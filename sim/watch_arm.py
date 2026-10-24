"""sim/watch.py with the B601-RS arm in the scene, every reach driven by the RL reach policy.

    .venv/bin/python -m sim.watch_arm --episodes 2 --out sim/watch_arm_rl.mp4

Per trial: the arm waits at the scan pose while the pile settles; the planner (rules, exact masks) picks;
the reach policy (sim/reach_policy.py) moves the tool to hover, down to the instrument, back up, over to the
staging spot (arm.yaml pick.staging_mm) and down; the instrument is welded to the gripper (kinematically)
between close and open. Requested instrument: picked and held up. The arm is kinematic (joint setpoints
followed exactly) and doesn't collide with the pile; the instruments are full physics.
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
from sim.reach_policy import ReachPolicy  # noqa: E402

BASE_WORLD = np.array([-0.30, -0.04, 0.0])     # arm base in the pile/camera frame: pile centre -> (300, 0) mm
CFG = load_arm_config()
STAGING_BASE = [(x * 1000, y * 1000) for x, y in CFG.pick.staging]          # mm, base frame
STAGING_WORLD = [(x + BASE_WORLD[0] * 1000, y + BASE_WORLD[1] * 1000) for x, y in STAGING_BASE]
HOVER = CFG.pick.table_z + CFG.pick.hover
GRIP_OPEN, GRIP_CLOSED = 0.05, 0.0


def _rot_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.ravel())
    return q


class ArmEpisode(watch.Episode):
    def __init__(self, rng, insts, templates, items, target, show, writer, policy: ReachPolicy):
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
        self._pose_arm()

    # ── arm ───────────────────────────────────────────────────────────────
    def _pose_arm(self) -> None:
        self.d.qpos[self.arm_adr] = self.q
        self.d.qpos[self.grip_adr] = self.grip
        self.d.qvel[self.m.jnt_dofadr[[self.m.joint("arm/" + n).id for n in ARM_JOINTS]]] = 0
        mujoco.mj_kinematics(self.m, self.d)

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
        """Policy rollout from the current joints to the base-frame pose; animate its own trajectory."""
        x, y = x_mm / 1000 - BASE_WORLD[0], y_mm / 1000 - BASE_WORLD[1]
        best = None
        for yaw in (np.radians(yaw_deg), np.radians(yaw_deg) + np.pi):     # pinch is symmetric
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
            q, steps, info = self.policy.rollout(x, y, z, yaw, self.q)
            if q is not None and (best is None or len(self.policy.path) < len(best)):
                best = list(self.policy.path)
        if best is None:
            self.caption = f"{self.caption_base}  -  reach policy did not converge ({what})"
            self.run_physics(1.0)
            return False
        self.caption = f"{self.caption_base}  -  RL reach: {what} ({len(best)} steps)"
        for q in best[::2] + [best[-1]]:
            self.q = q
            self._pose_arm()
            if not self.run_physics(1.0 / watch.FPS):
                return False
        return True

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

    def pick(self, i: int, p: dict) -> bool:
        x, y, yaw = p["x_mm"], p["y_mm"], p["yaw_deg"]
        z_grasp = CFG.pick.table_z + max(p["thickness_mm"] / 2000.0, 0.002)
        ok = (self.reach(x, y, HOVER, yaw, "hover over it") and self.gripper(GRIP_OPEN)
              and self.reach(x, y, z_grasp, yaw, "down to grasp"))
        if not ok:
            return False
        self.gripper(GRIP_CLOSED)
        self.weld(i)
        return self.reach(x, y, HOVER, yaw, "lift")

    def place(self, xy_world: tuple[float, float], thickness_mm: float) -> bool:
        yaw = np.degrees(CFG.pick.staging_yaw)
        ok = (self.reach(*xy_world, HOVER, yaw, "carry to staging")
              and self.reach(*xy_world, CFG.pick.table_z + thickness_mm / 2000.0 + 0.005, yaw, "set down"))
        self.welded = None
        self.gripper(GRIP_OPEN)
        return ok and self.reach(*xy_world, HOVER, yaw, "back up")

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
            if not self.pick(i, p["target"]):
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
    ap.add_argument("--policy", default=str(Path(__file__).resolve().parent / "policy" / "reach_policy.zip"))
    ap.add_argument("--out", default="sim/watch_arm_rl.mp4")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args(argv)
    insts = load_instruments()
    templates = build_templates(insts, 0.5)
    rng = np.random.default_rng(args.seed)
    policy = ReachPolicy(args.policy, max_steps=200)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"avc1"), watch.FPS, (2 * watch.W, watch.H_PX))
    for _ in range(args.episodes):
        items = [str(f) for f in rng.choice(list(insts), size=int(rng.integers(5, 9)), replace=False)]
        target = str(rng.choice(items))
        res = ArmEpisode(rng, insts, templates, items, target, not args.no_window, writer, policy).play()
        print(f"{display_name(target)}: {res}", flush=True)
        if res == "quit":
            break
    writer.release()
    print(f"video: {args.out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
