"""Camera-guided (closed-loop) reach: the policy sees the target only through the simulated wrist camera.

Scene: a real instrument (instruments.yaml) settled on the table by sim/pile_sim.py, with 0-2 others dropped
under it, then moved to a random spot in the top-down workspace. Its grasp point and jaw yaw are what
scrub_vision.identify() reads off an exact top-down mask (template grasp point, yaw = axis + 90 deg).
Goal: tool at that grasp point, `h` above the table (40-120 mm, an input: the contact descent does the rest),
jaws along the yaw (either way round: the pinch is symmetric).
Observation (every step, from the camera, never the target's world pose):
  [visible, grasp-point pixel offset from the fingertips (u/W, v/H), sin/cos 2*(target axis - jaw axis) in
   the image, apparent length / W, fingertip pixel (u/W, v/H), joints / pi, previous action, h / 0.1 m,
   the 3 gripper LANDMARKS' pixels (u/W, v/H; always fresh, 3 px jitter)].
  Perception-like noise: 3 px jitter, 3 deg angle jitter, 7 % dropped frames, and the target is hidden while
  its grasp point is behind the fingers (below the fingertip pixel, +-0.2 W); hidden -> last features held,
  visible = 0.
Mount: arm.yaml wrist_camera, randomised per episode (mount_noise=1: +-10 mm back/side, +-5 deg tilt;
scaled by mount_noise, 0 = nominal).
Action: joint deltas MAX_DQ * a^3 (a in [-1, 1]): 16 mm at the tool for a=1, sub-mm for |a| < 0.2. A
true-error-adaptive scale (sim/reach_env.py) would leak the hidden goal, so the cube gives the coarse/fine
range instead.
Reward: sim/reach_env.py's, on the TRUE pose error. Success: true tool pose within 3 mm / 3 deg.
Start: IK pose near the goal (up to 80 mm off, 0-60 mm higher, +-45 deg yaw): the scan already put the arm
roughly above the instrument; the camera closes the loop from there. 150-step episodes, no termination.
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np

from scrub_arm.config import load_arm_config
from scrub_arm.kinematics import Kinematics
from scrub_vision.identify import _apply, identify
from scrub_vision.render import render_scene, scene_mesh

from sim.arm_model import load_model, set_arm, tool_pose, touches_table
from sim.declutter_eval import CAM, H, _assets
from sim.pile_sim import settle_pile
from sim.reach_env import POS_TOL, ROT_TOL_DEG
from sim.wrist_cam import CAM_H, CAM_W, NOMINAL, WristCam

MAX_DQ = 0.04
N_OBS = 27
# Gripper landmarks (tool frame, m) the camera always sees at fixed pixels (fiducials / finger corners):
# between the fingertips, a marker 40 mm back on the finger top, a fingertip corner 20 mm along the jaws.
# Their pixels depend only on the camera mount, so they let a controller calibrate the mount online.
LANDMARKS = np.array([[0.0, 0.0, 0.0], [-0.04, 0.0, 0.019], [0.0, 0.02, 0.0]])
FINGERS_HALF_W, FINGERS_ABOVE_TIP = 0.2 * CAM_W, 10.0       # px: the fingers hide what is behind/below the tips


def make_target(rng: np.random.Generator) -> dict:
    """Settle 1-3 instruments (the target dropped last, on top) and read the target as identify() does."""
    insts, templates = _assets()
    while True:
        items = [str(f) for f in rng.choice(list(insts), size=int(rng.integers(1, 4)), replace=False)]
        pile, _ = settle_pile(items, rng)
        _, masks = render_scene([(scene_mesh(insts[f]), T) for f, T in pile], CAM, rng)
        if masks[-1].sum() < 300:
            continue
        d = identify(masks[-1], None, H, templates, insts)
        if not d.candidates:
            continue
        return {"fid": items[-1], "pile": pile, "grasp": np.array(d.grasp_mm) / 1000,
                "centroid": _apply(H, d.centroid_px)[0] / 1000, "axis": np.radians(d.axis_deg),
                "yaw": np.radians(d.yaw_deg), "length": d.length_mm / 1000, "thickness": d.thickness_mm / 1000}


def placed(t: dict, phi: float, at: np.ndarray, table_z: float) -> dict:
    """The target rotated by phi about its grasp point and moved so the grasp point is at `at` (base frame)."""
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    move = lambda p: R @ (np.asarray(p) - t["grasp"]) + at  # noqa: E731
    return {**t, "grasp": at.copy(), "centroid": move(t["centroid"]), "axis": t["axis"] + phi, "yaw": t["yaw"] + phi,
            "z": table_z + t["thickness"] / 2, "phi": phi, "at": at.copy(), "grasp_pile": t["grasp"]}


class VisualReachEnv(gym.Env):
    OBS_VISIBLE = 0

    def __init__(self, seed: int | None = None, noise: bool = True, mount_noise: float = 1.0, nudge: bool = False,
                 max_steps: int = 150, pool_size: int = 64, table_penalty: float = 5.0,
                 limit_penalty: float = 0.1) -> None:
        cfg = load_arm_config()
        self.lo, self.hi, self.table_z = cfg.lower, cfg.upper, cfg.pick.table_z
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.cam = WristCam(self.model)
        self.kin = Kinematics()
        self.noise, self.mount_noise, self.nudge = noise, mount_noise, nudge
        self.dropout = 0.07 if noise else 0.0
        self.max_steps, self.pool_size = max_steps, pool_size
        self.table_penalty, self.limit_penalty = table_penalty, limit_penalty
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_OBS,), np.float32)
        self._rng = np.random.default_rng(seed)
        self.pool: list[dict] = []

    # ── episode setup ─────────────────────────────────────────────────────
    def _ik(self, x, y, z, yaw, seed_q):
        return self.kin.ik_topdown(x, y, z, yaw, seed_q, (self.lo, self.hi), allow_flip=True)

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        if options and "target" in options:
            base = options["target"]
        else:
            if len(self.pool) < self.pool_size:
                self.pool.append(make_target(rng))
            base = self.pool[int(rng.integers(len(self.pool)))]
        while True:
            r, a = rng.uniform(0.17, 0.34), rng.uniform(-2 * np.pi / 3, 2 * np.pi / 3)
            t = placed(base, rng.uniform(-np.pi, np.pi), np.array([r * np.cos(a), r * np.sin(a)]), self.table_z)
            h = rng.uniform(0.04, 0.12)
            goal = (*t["grasp"], self.table_z + h, t["yaw"])
            q_goal = self._ik(*goal, np.radians([0, 45, 70, 0, 65, 0]))
            off = rng.uniform(0, 0.08) * np.array([np.cos(b := rng.uniform(0, 2 * np.pi)), np.sin(b)])
            q0 = None if q_goal is None else self._ik(goal[0] + off[0], goal[1] + off[1], goal[2] + rng.uniform(0, 0.06),
                                                      goal[3] + rng.uniform(-np.pi / 4, np.pi / 4), q_goal)
            if q0 is not None:
                break
        mount = dict(NOMINAL)
        if self.mount_noise:
            k = float(self.mount_noise)                      # 1 = +-10 mm / +-5 deg (uncalibrated)
            mount = {"back": mount["back"] + k * rng.uniform(-0.01, 0.01),
                     "side": mount["side"] + k * rng.uniform(-0.01, 0.01),
                     "tilt_deg": mount["tilt_deg"] + k * rng.uniform(-5, 5)}
        self.cam.set_mount(self.model, **mount)
        self.target, self.h, self.goal, self.q = t, h, goal, q0
        self.steps, self.prev_a, self.last = 0, np.zeros(6), np.zeros(7)
        self.nudge_at, self.push = -1, None
        if self.nudge:                                 # drawn here so every method on this episode gets the same push
            phi = rng.uniform(0, 2 * np.pi)
            self.nudge_at = int(rng.integers(30, 81))
            self.push = (rng.uniform(0.010, 0.030) * np.array([np.cos(phi), np.sin(phi)]),
                         rng.uniform(-np.radians(15), np.radians(15)))
        set_arm(self.model, self.data, self.q)
        return self._obs(), {"goal": goal, "q_goal": q_goal, "fid": t["fid"], "mount": mount}

    def grasp_world(self) -> np.ndarray:
        return np.array([*self.target["grasp"], self.target["z"]])

    # ── camera ────────────────────────────────────────────────────────────
    def features(self, est: dict | None = None) -> dict:
        """What the wrist camera reports about the target (`est`: a perception estimate, {"hidden": True} if
        perception lost it; default: the ground truth)."""
        if est is not None and est.get("hidden"):
            return {"visible": False}
        t, rng = est or self.target, self._rng
        c, a, L, z = np.asarray(t["centroid"]), t["axis"], t["length"], t["z"]
        e = np.array([np.cos(a), np.sin(a)]) * L / 2
        tip = self.data.site("tool").xpos
        jaw = self.data.site("tool").xmat.reshape(3, 3) @ self.kin.jaw_axis_local
        uv = self.cam.project(self.data, np.array([[*t["grasp"], z], [*(c - e), z], [*(c + e), z], tip, tip + 0.02 * jaw]))
        if self.noise:
            uv[:3] += rng.normal(0, 3.0, (3, 2))
        g = uv[0]
        visible = bool(np.all(np.isfinite(uv)) and 0 <= g[0] < CAM_W and 0 <= g[1] < CAM_H
                       and not (abs(g[0] - uv[3][0]) < FINGERS_HALF_W and g[1] > uv[3][1] - FINGERS_ABOVE_TIP)
                       and rng.random() >= self.dropout)
        if not visible:
            return {"visible": False}
        d_t, d_j = uv[2] - uv[1], uv[4] - uv[3]
        rel = np.arctan2(d_t[1], d_t[0]) - np.arctan2(d_j[1], d_j[0]) + (rng.normal(0, np.radians(3)) if self.noise else 0)
        return {"visible": True, "offset": [(g[0] - uv[3][0]) / CAM_W, (g[1] - uv[3][1]) / CAM_H],
                "rel": [np.sin(2 * rel), np.cos(2 * rel)], "length": np.linalg.norm(d_t) / CAM_W,
                "tip": [uv[3][0] / CAM_W, uv[3][1] / CAM_H]}

    def _obs(self, est: dict | None = None) -> np.ndarray:
        f = self.features(est)
        if f["visible"]:
            self.last = np.array([*f["offset"], *f["rel"], f["length"], *f["tip"]])
        s = self.data.site("tool")
        lm = self.cam.project(self.data, s.xpos + LANDMARKS @ s.xmat.reshape(3, 3).T)
        if self.noise:
            lm += self._rng.normal(0, 3.0, lm.shape)
        return np.concatenate([[float(f["visible"])], self.last, self.q / np.pi, self.prev_a, [self.h / 0.1],
                               (lm / [CAM_W, CAM_H]).ravel()]).astype(np.float32)

    # ── dynamics ──────────────────────────────────────────────────────────
    def errors(self) -> tuple[float, float]:
        p, R = tool_pose(self.model, self.data)
        d = float(np.linalg.norm(p - np.array(self.goal[:3])))
        th = min(float(np.degrees(np.arccos(np.clip((np.trace(self.kin._target_rotation(y).T @ R) - 1) / 2, -1, 1))))
                 for y in (self.goal[3], self.goal[3] + np.pi))
        return d, th

    def step(self, action, perceive=None):
        """perceive(env) -> target estimate for this step's observation (sim/eval_servo.py); default ground truth."""
        a = np.clip(np.asarray(action, float), -1, 1)
        self.q = np.clip(self.q + MAX_DQ * a ** 3, self.lo, self.hi)
        self.prev_a, self.steps = a, self.steps + 1
        info = {}
        if self.steps == self.nudge_at:                # pushed 10-30 mm and turned up to 15 deg about its grasp point
            t, (dxy, rot) = self.target, self.push
            c, s = np.cos(rot), np.sin(rot)
            cen = t["grasp"] + np.array([[c, -s], [s, c]]) @ (t["centroid"] - t["grasp"])
            self.target = {**t, "grasp": t["grasp"] + dxy, "centroid": cen + dxy, "axis": t["axis"] + rot,
                           "yaw": t["yaw"] + rot}
            self.goal = (self.goal[0] + dxy[0], self.goal[1] + dxy[1], self.goal[2], self.goal[3] + rot)
            info["nudged"] = True
        set_arm(self.model, self.data, self.q)
        obs = self._obs(perceive(self) if perceive else None)
        d, th = self.errors()
        table = touches_table(self.model, self.data)
        pinned = int(np.sum((self.q <= self.lo + 1e-6) | (self.q >= self.hi - 1e-6)))
        success = d < POS_TOL and th < ROT_TOL_DEG and not table
        r = (-d / 0.01 - th / 10 - 0.5 * np.log10(d * 1000 + 0.1) - 0.5 * np.log10(th + 0.1) + float(success)
             - self.limit_penalty * pinned - (self.table_penalty if table else 0.0))
        info.update(success=success, pos_err=d, rot_err_deg=th, table=table, steps=self.steps)
        return obs, float(r), False, self.steps >= self.max_steps, info


def evaluate(model, n: int = 200, seed: int = 777, env_cls=None, rows: list | None = None, **env_kw) -> dict:
    """Deterministic policy on n fixed held-out episodes. Scored at the LAST step (a deployment can't see the
    true error, so it can't stop at the first good step); `reached` = within tolerance at any step.
    model: anything with predict(obs, deterministic) (SAC, sim.ibvs.IBVS); env_cls: VisualReachEnv or
    sim.ibvs.ResidualEnv. rows: if given, per-episode results are appended to it."""
    env = (env_cls or VisualReachEnv)(seed=seed, **env_kw)
    rng = np.random.default_rng(seed)
    targets = [make_target(rng) for _ in range(n)]
    pe, re_, ok, first = [], [], [], []
    for i, t in enumerate(targets):
        obs, _ = env.reset(seed=seed + 1 + i, options={"target": t})
        if hasattr(model, "est"):                       # sim.ibvs.IBVS keeps per-episode state
            model.reset()
        hit = None
        for _ in range(env.max_steps):
            obs, _, _, _, info = env.step(model.predict(obs, deterministic=True)[0])
            if info["success"] and hit is None:
                hit = info["steps"]
        pe.append(info["pos_err"] * 1000)
        re_.append(info["rot_err_deg"])
        ok.append(info["success"])
        first.append(hit)
        if rows is not None:
            rows.append({"fid": t["fid"], "success": bool(info["success"]), "pos_mm": pe[-1], "rot_deg": re_[-1],
                         "reached_at": hit})
    pe, hits = np.array(pe), [h for h in first if h is not None]
    return {"success": float(np.mean(ok)), "reached": len(hits) / n, "pos_mm_median": float(np.median(pe)),
            "pos_mm_p95": float(np.percentile(pe, 95)), "rot_deg_median": float(np.median(re_)),
            "steps_median_success": float(np.median(hits)) if hits else None}
