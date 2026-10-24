"""sim/watch.py with the B601-RS arm in the scene, reaching with the RL policies and the wrist camera.

    .venv/bin/python -m sim.watch_arm --episodes 2 --camera sim/policy/reach_R1_best.zip --out sim/watch_arm_rl.mp4

Three panels: MuJoCo physics | the live simulated wrist camera with the camera policy's tracked target |
the scan view and the planner's decision. Per trial: the arm waits at the scan pose while the pile settles;
the planner (rules, exact masks) picks; the blind reach policy (sim/reach_policy.py) brings the tool roughly
over the instrument, the camera-guided policy (sim/ibvs.py ResidualEnv: IBVS + learned residual, wrist camera
features only) closes in to hover and then 40 mm above it, a straight contact descent (IK, as on the real
arm) goes down to grasp; the instrument is welded to the gripper (kinematically) between close and open;
the blind policy carries it to the staging spot (arm.yaml pick.staging_mm). Requested instrument: picked
and held up. The arm is kinematic (joint setpoints followed exactly) and doesn't collide with the pile; the
instruments are full physics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "perception"), str(ROOT / "arm")]

from scrub_arm.config import load_arm_config  # noqa: E402
from scrub_vision.identify import annotate  # noqa: E402
from scrub_vision.planner import display_name, plan  # noqa: E402
from scrub_vision.render import scene_mesh  # noqa: E402
from scrub_vision.templates import build_templates, interchangeable, load_instruments  # noqa: E402

from sim import watch  # noqa: E402
from sim.arm_model import ARM_JOINTS, XML  # noqa: E402
from sim.reach_env import SCAN  # noqa: E402
from sim.arm_model import set_arm  # noqa: E402
from sim.ibvs import ResidualEnv  # noqa: E402
from sim.reach_policy import ReachPolicy  # noqa: E402
from sim.wrist_cam import CAM_H, CAM_W, WristCam  # noqa: E402
from scrub_vision.identify import _apply  # noqa: E402

BASE_WORLD = np.array([-0.30, -0.04, 0.0])     # arm base in the pile/camera frame: pile centre -> (300, 0) mm
CFG = load_arm_config()
STAGING_BASE = [(x * 1000, y * 1000) for x, y in CFG.pick.staging]          # mm, base frame
STAGING_WORLD = [(x + BASE_WORLD[0] * 1000, y + BASE_WORLD[1] * 1000) for x, y in STAGING_BASE]
HOVER = CFG.pick.table_z + CFG.pick.hover
GRIP_OPEN, GRIP_CLOSED = 0.05, 0.0
PW, PH, BAR = 640, 360, 44                      # video panel size and caption bar


def _rot_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


class ArmEpisode(watch.Episode):
    def __init__(self, rng, insts, templates, items, target, show, writer, policy: ReachPolicy, camera):
        super().__init__(rng, insts, templates, items, target, show, writer)
        spec = mujoco.MjSpec.from_string(self.xml)
        arm = mujoco.MjSpec.from_file(str(XML))
        arm.delete(arm.geom("table"))
        for g in arm.geoms:
            g.contype, g.conaffinity = 0, 0              # kinematic arm: it doesn't shove the pile around
        for l in arm.lights:
            arm.delete(l)
        spec.worldbody.add_frame(pos=BASE_WORLD).attach_body(arm.body("base_link"), "arm/", "")
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.r.close()                                   # the parent's renderer, for the pile-only model
        self.r = mujoco.Renderer(self.m, watch.H_PX, watch.W)
        self.cam.lookat[:] = (-0.02, 0.0, 0.12)
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 1.05, -35, 140
        self.arm_adr = np.array([self.m.jnt_qposadr[self.m.joint("arm/" + n).id] for n in ARM_JOINTS])
        self.grip_adr = [self.m.jnt_qposadr[self.m.joint(f"arm/gripper_joint{k}").id] for k in (1, 2)]
        self.policy, self.q, self.grip, self.welded = policy, SCAN.copy(), GRIP_OPEN, None
        self.camera, self.cenv = camera, ResidualEnv(noise=True, mount_noise=0.0, pool_size=1)
        self.wr = mujoco.Renderer(self.m, CAM_H, CAM_W)
        self.wcam, self.track = WristCam(self.m, "arm/wrist"), None
        self._pose_arm()

    # ── arm ───────────────────────────────────────────────────────────────
    def _pose_arm(self) -> None:
        self.d.qpos[self.arm_adr] = self.q
        self.d.qpos[self.grip_adr] = self.grip
        self.d.qvel[self.m.jnt_dofadr[[self.m.joint("arm/" + n).id for n in ARM_JOINTS]]] = 0
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)

    def frame(self) -> bool:
        """physics | live wrist camera + tracked target | scan view + decision."""
        self.r.update_scene(self.d, self.cam)
        left = cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)
        self.wr.update_scene(self.d, camera="arm/wrist")
        mid = np.ascontiguousarray(cv2.cvtColor(self.wr.render(), cv2.COLOR_RGB2BGR))
        if self.track is not None:                   # the camera policy's grasp estimate (IBVS), world frame
            xy, yaw = self.track["xy"] + BASE_WORLD[:2], self.track["yaw"] - np.pi / 2
            d = 0.03 * np.array([np.cos(yaw), np.sin(yaw), 0])
            uv = self.wcam.project(self.d, np.array([[*xy, 0.002], [*xy, 0.002] + d, [*xy, 0.002] - d]))
            if np.all(np.isfinite(uv)):
                p = [tuple(int(v) for v in u) for u in uv]
                cv2.line(mid, p[1], p[2], (0, 255, 255), 2)
                cv2.circle(mid, p[0], 9, (0, 0, 255), 2)
        panels = [cv2.resize(x, (PW, PH)) for x in (left, mid, self.right)]
        for x, label in zip(panels, ("physics (MuJoCo)", "wrist camera (sim) + tracked grasp point",
                                     "scan + decision")):
            cv2.putText(x, label, (10, PH - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        out = np.vstack([np.full((BAR, 3 * PW, 3), 25, np.uint8), np.hstack(panels)])
        cv2.putText(out, self.caption[:150], (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.writer is not None:
            self.writer.write(out)
        if self.show:
            cv2.imshow("scrub arm: simulation", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    def run_physics(self, seconds: float, hold: dict | None = None) -> bool:
        hold = dict(hold or {})
        if self.welded is not None:
            i, rel = self.welded
            s = self.d.site("arm/tool")
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = s.xmat.reshape(3, 3), s.xpos
            W = T @ rel
            hold[i] = np.r_[W[:3, 3], _rot_quat(W[:3, :3])]
        steps = int(1.0 / watch.FPS / self.m.opt.timestep)
        for _ in range(int(round(seconds * watch.FPS))):
            for _ in range(steps):
                self.d.qpos[self.arm_adr] = self.q
                self.d.qpos[self.grip_adr] = self.grip
                for i, q in hold.items():
                    self.d.qpos[self.adr[i]:self.adr[i] + 7] = q
                    self.d.qvel[self.vadr[i]:self.vadr[i] + 6] = 0
                mujoco.mj_step(self.m, self.d)
            if not self.frame():
                return False
        return True

    def reach(self, x_mm: float, y_mm: float, z: float, yaw_deg: float, what: str) -> bool:
        """Blind policy rollout from the current joints to a world pose; animate its trajectory either way."""
        x, y = x_mm / 1000 - BASE_WORLD[0], y_mm / 1000 - BASE_WORLD[1]
        runs = []
        for yaw in (np.radians(yaw_deg), np.radians(yaw_deg) + np.pi):     # pinch is symmetric
            q, _, info = self.policy.rollout(x, y, z, (yaw + np.pi) % (2 * np.pi) - np.pi, self.q)
            runs.append((q is not None, info.get("pos_err", 1.0), list(self.policy.path)))
        ok, err, path = max(runs, key=lambda r: (r[0], -r[1]))
        self.caption = (f"{self.caption_base}  -  blind RL reach: {what} "
                        f"({'converged' if ok else f'stopped {err * 1000:.0f} mm off'})")
        for q in path[::2] + path[-1:]:
            self.q = q
            self._pose_arm()
            if not self.run_physics(1.0 / watch.FPS):
                return False
        return True

    def servo(self, est: dict, h: float, what: str) -> bool:
        """Camera-guided policy (IBVS + learned residual) on wrist-camera features of the scanned instrument."""
        env = self.cenv
        env.target, env.h, env.q = est, h, self.q.copy()
        env.goal = (*est["grasp"], CFG.pick.table_z + h, est["yaw"])
        env.steps, env.prev_a, env.last, env.nudge_at = 0, np.zeros(6), np.zeros(7), -1
        env.ctl.reset()
        set_arm(env.model, env.data, env.q)
        obs = env._with_base(env._obs())
        for k in range(env.max_steps):
            obs, _, _, _, info = env.step(self.camera.predict(obs, deterministic=True)[0])
            self.q, self.track = env.q.copy(), env.ctl.est
            if k % 2:
                continue
            self.caption = (f"{self.caption_base}  -  camera-guided RL: {what}, step {k + 1}, "
                            f"target {'seen' if obs[0] > 0.5 else 'hidden'}")
            self._pose_arm()
            if not self.run_physics(1.0 / watch.FPS):
                return False
        return True

    def ik_move(self, xy_world, z: float, yaw: float, what: str) -> bool:
        """Straight vertical move at the current spot (the contact descent / lift, IK as on the real arm)."""
        x, y = xy_world[0] - BASE_WORLD[0], xy_world[1] - BASE_WORLD[1]
        q = self.cenv.kin.ik_topdown(x, y, z, yaw, self.q, (self.cenv.lo, self.cenv.hi))
        if q is None:
            return False
        self.caption = f"{self.caption_base}  -  {what}"
        return self.joint_move(q, 1.0)

    def joint_move(self, q_goal: np.ndarray, seconds: float = 1.5) -> bool:
        q0 = self.q.copy()
        for t in np.linspace(0, 1, int(seconds * watch.FPS)):
            self.q = q0 + (q_goal - q0) * (3 * t * t - 2 * t * t * t)
            if not self.run_physics(1.0 / watch.FPS):
                return False
        return True

    def gripper(self, value: float) -> bool:
        g0 = self.grip
        for t in np.linspace(0, 1, 10):
            self.grip = g0 + (value - g0) * t
            if not self.run_physics(1.0 / watch.FPS):
                return False
        return True

    def weld(self, i: int) -> None:
        s, a = self.d.site("arm/tool"), self.adr[i]
        T_tool, T_obj = np.eye(4), np.eye(4)
        T_tool[:3, :3], T_tool[:3, 3] = s.xmat.reshape(3, 3), s.xpos
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.d.qpos[a + 3:a + 7])
        T_obj[:3, :3], T_obj[:3, 3] = R.reshape(3, 3), self.d.qpos[a:a + 3]
        self.welded = (i, np.linalg.inv(T_tool) @ T_obj)

    def scanned(self, det) -> dict:
        """The scan's estimate of an instrument (perception frame -> arm base frame), for the camera policy."""
        c = _apply(watch.H, det.centroid_px)[0] / 1000
        return {"grasp": np.array(det.grasp_mm) / 1000 - BASE_WORLD[:2], "centroid": c - BASE_WORLD[:2],
                "axis": np.radians(det.axis_deg), "yaw": np.radians(det.yaw_deg), "length": det.length_mm / 1000,
                "z": CFG.pick.table_z + det.thickness_mm / 2000.0}

    def pick(self, i: int, det) -> bool:
        est = self.scanned(det)
        here = tuple(est["grasp"] * 1000 + BASE_WORLD[:2] * 1000)
        ok = (self.reach(*here, HOVER, det.yaw_deg, "roughly over it") and self.gripper(GRIP_OPEN)
              and self.servo(est, CFG.pick.hover, "hover over it") and self.servo(est, 0.04, "40 mm above it"))
        if not ok:
            return False
        p, R = self.cenv.kin.fk(self.q)
        yaw_now = np.arctan2(*(R @ self.cenv.kin.jaw_axis_local)[1::-1])
        xy = p[:2] + BASE_WORLD[:2]
        z_grasp = CFG.pick.table_z + max(det.thickness_mm / 2000.0, 0.002)
        if not self.ik_move(xy, z_grasp, yaw_now, "contact descent"):
            return False
        self.gripper(GRIP_CLOSED)
        self.weld(i)
        self.track = None
        return self.ik_move(xy, HOVER, yaw_now, "lift")

    def place(self, xy_world: tuple[float, float], thickness_mm: float) -> bool:
        yaw = np.degrees(CFG.pick.staging_yaw)
        ok = self.reach(*xy_world, HOVER, yaw, "carry to staging")
        p, R = self.cenv.kin.fk(self.q)
        xy = p[:2] + BASE_WORLD[:2]
        yaw_now = np.arctan2(*(R @ self.cenv.kin.jaw_axis_local)[1::-1])
        ok = ok and self.ik_move(xy, CFG.pick.table_z + thickness_mm / 2000.0 + 0.005, yaw_now, "set down")
        self.welded = None
        self.gripper(GRIP_OPEN)
        return ok and self.ik_move(xy, HOVER, yaw_now, "back up")

    def play(self, max_moves: int = 6) -> str:
        cx, cy = watch.PILE_CENTER_MM
        hx, hy = watch.PILE_HALF_MM
        for i in range(len(self.items)):
            a = self.adr[i]
            self.d.qpos[a:a + 3] = (self.rng.uniform(cx - hx * .7, cx + hx * .7) / 1000,
                                    self.rng.uniform(cy - hy * .7, cy + hy * .7) / 1000, 0.03 + 0.02 * i)
            self.d.qpos[a + 3:a + 7] = watch._quat(self.rng.uniform(-np.pi, np.pi), bool(self.rng.random() < .5))
        self.caption_base = f"Request: {display_name(self.target)}"
        self.caption = f"{self.caption_base}  -  dropping {len(self.items)} instruments; arm at scan pose"
        if not self.run_physics(2.5):
            return "quit"
        used, moves = set(), 0
        while True:
            bgr, dets, truth = self.look()
            p = plan(dets, self.target, watch.H, STAGING_WORLD, used)
            chosen = None
            if p.get("target"):
                chosen = min(dets, key=lambda d: np.hypot(d.grasp_mm[0] - p["target"]["x_mm"],
                                                          d.grasp_mm[1] - p["target"]["y_mm"]))
            self.right = cv2.resize(annotate(bgr, dets, chosen), (watch.W, watch.H_PX))
            tag = {"grasp_target": "GRASP TARGET", "move_aside": "MOVE ASIDE", "fail": "GIVE UP"}[p["action"]]
            self.caption = f"{self.caption_base}  -  {tag}: {p['say']}"
            if not self.run_physics(1.2):
                return "quit"
            if p["action"] == "fail":
                return "fail"
            i = truth[id(chosen)]
            if not self.pick(i, chosen):
                return "reach_failed"
            if p["action"] == "grasp_target":
                ok = self.items[i] in interchangeable(self.insts, self.target)
                self.caption = (f"{self.caption_base}  -  "
                                f"{'GOT IT' if ok else 'WRONG: ' + display_name(self.items[i])} after {moves} move(s)")
                self.run_physics(1.5)
                return "grasped" if ok else "wrong"
            used.add(p["staging_index"])
            if not self.place(STAGING_WORLD[p["staging_index"]], p["target"]["thickness_mm"]):
                return "reach_failed"
            self.staged.add(i)
            self.caption = f"{self.caption_base}  -  back to scan pose"
            if not self.joint_move(SCAN):
                return "quit"
            moves += 1
            if moves >= max_moves:
                return "cap"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sim.watch_arm")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--policy", default=str(Path(__file__).resolve().parent / "policy" / "reach_B_best.zip"),
                    help="blind reach policy (approach and carry)")
    ap.add_argument("--camera", required=True, help="camera-guided residual policy (sim.train_reach --env residual)")
    ap.add_argument("--out", default="sim/watch_arm_rl.mp4")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args(argv)
    insts = load_instruments()
    templates = build_templates(insts, 0.5)
    rng = np.random.default_rng(args.seed)
    from stable_baselines3 import SAC

    policy = ReachPolicy(args.policy, max_steps=200)
    camera = SAC.load(args.camera, env=ResidualEnv(pool_size=1), device="cpu")
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"avc1"), watch.FPS, (3 * PW, PH + BAR))
    for _ in range(args.episodes):
        items = [str(f) for f in rng.choice(list(insts), size=int(rng.integers(5, 9)), replace=False)]
        target = str(rng.choice(items))
        res = ArmEpisode(rng, insts, templates, items, target, not args.no_window, writer, policy, camera).play()
        print(f"{display_name(target)}: {res}", flush=True)
        if res == "quit":
            break
    writer.release()
    print(f"video: {args.out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
