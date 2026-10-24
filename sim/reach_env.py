"""Reach-only RL environment: drive the B601-RS tool frame to a top-down pose (x, y, z, jaw yaw).

Goal-conditioned (gymnasium-robotics style, for stable-baselines3 HER): obs = {observation, achieved_goal,
desired_goal}; a goal is the tool position (decimetres, base frame) + the first two columns of its rotation.
Kinematic MuJoCo rollout (the real servos track position setpoints, so a step = a new setpoint):
action in [-1, 1]^6 -> joint delta a * MAX_DQ * step_scale(errors), clipped to arm.yaml limits; the scale
(coarse far away, ~0.5 mm at the tool when close) is part of the observation.
Reward (compute_reward): -(pos err / 1 cm) - (rot err / 10 deg) - 0.5 log10(err + 0.1) per term (steep near
zero), +1 per step inside 3 mm and 3 deg, minus info["penalty"] (joints pinned at limits, table contact).
No termination on success: fixed 150-step episodes (HER relabelling assumes it); a deployment stops at
the first step inside the tolerance (sim/reach_policy.py).
Targets: ~uniform over the top-down workspace (r 120-400 mm from the base, +-135 deg around it,
z table+5 .. table+125 mm, any yaw), kept only if scrub_arm Kinematics.ik_topdown reaches them.
Start: the scan pose half the time, else the IK solution for another random target.
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np

from scrub_arm.config import load_arm_config
from scrub_arm.kinematics import Kinematics
from sim.arm_model import load_model, set_arm, tool_pose, touches_table

MAX_DQ = 0.04          # rad per step at full scale
POS_TOL, ROT_TOL_DEG = 3e-3, 3.0
SCAN = np.radians([0.0, 45.0, 70.0, 0.0, 65.0, 0.0])   # arm/config/poses.yaml scan


def step_scale(d: float, th_deg: float) -> float:
    """Action multiplier from the current errors: 1 beyond 50 mm / 20 deg, down to 0.03 near zero."""
    return float(np.clip(max(d / 0.05, th_deg / 20.0), 0.03, 1.0))


def goal(p: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.concatenate([p * 10, R[:, 0], R[:, 1]]).astype(np.float32)


def goal_errors(achieved, desired) -> tuple[np.ndarray, np.ndarray]:
    """(position error m, rotation error deg), batched over leading dims."""
    a, g = np.asarray(achieved, np.float64), np.asarray(desired, np.float64)
    d = np.linalg.norm(a[..., :3] - g[..., :3], axis=-1) / 10

    def rot(x):
        c0 = x[..., 3:6] / np.linalg.norm(x[..., 3:6], axis=-1, keepdims=True)
        c1 = x[..., 6:9] - np.sum(x[..., 6:9] * c0, -1, keepdims=True) * c0
        c1 = c1 / np.linalg.norm(c1, axis=-1, keepdims=True)
        return np.stack([c0, c1, np.cross(c0, c1)], axis=-1)

    tr = np.einsum("...ij,...ij->...", rot(a), rot(g))         # trace(Ra^T Rg)
    return d, np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


class ReachEnv(gym.Env):
    def __init__(self, seed: int | None = None, max_steps: int = 150, table_penalty: float = 5.0,
                 limit_penalty: float = 0.1) -> None:
        cfg = load_arm_config()
        self.lo, self.hi, self.table_z = cfg.lower, cfg.upper, cfg.pick.table_z
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.kin = Kinematics()
        self.max_steps, self.table_penalty, self.limit_penalty = max_steps, table_penalty, limit_penalty
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), np.float32)  # noqa: E731
        self.observation_space = gym.spaces.Dict({"observation": box(13), "achieved_goal": box(9),
                                                  "desired_goal": box(9)})
        self._rng = np.random.default_rng(seed)

    # ── targets ───────────────────────────────────────────────────────────
    def sample_target(self, rng: np.random.Generator):
        """(x, y, z, yaw) and its ik_topdown solution, resampled until reachable."""
        while True:
            r, a = rng.uniform(0.12, 0.40), rng.uniform(-0.75 * np.pi, 0.75 * np.pi)
            t = (r * np.cos(a), r * np.sin(a), self.table_z + rng.uniform(0.005, 0.125), rng.uniform(-np.pi, np.pi))
            q = self.kin.ik_topdown(*t, SCAN, (self.lo, self.hi), allow_flip=False)
            if q is not None:
                return t, q

    def set_target(self, target, q0) -> dict:
        self.target = tuple(float(v) for v in target)
        self.desired = goal(np.array(self.target[:3]), self.kin._target_rotation(self.target[3]))
        self.q, self.steps = np.clip(np.asarray(q0, float), self.lo, self.hi), 0
        return self._obs()

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        rng = self._rng
        q0 = SCAN if rng.random() < 0.5 else self.sample_target(rng)[1]
        target, q_ik = self.sample_target(rng)
        return self.set_target(target, q0), {"target": self.target, "q_ik": q_ik}

    # ── dynamics ──────────────────────────────────────────────────────────
    def _obs(self) -> dict:
        set_arm(self.model, self.data, self.q)
        p, R = tool_pose(self.model, self.data)
        self.achieved = goal(p, R)
        d, th = goal_errors(self.achieved, self.desired)
        self.scale = step_scale(float(d), float(th))
        return {"observation": np.concatenate([self.q / np.pi, [self.scale], p * 10, R[:, 2]]).astype(np.float32),
                "achieved_goal": self.achieved.copy(), "desired_goal": self.desired.copy()}

    def compute_reward(self, achieved_goal, desired_goal, info):
        d, th = goal_errors(achieved_goal, desired_goal)
        pen = (np.array([i.get("penalty", 0.0) for i in info]) if isinstance(info, (list, tuple, np.ndarray))
               else info.get("penalty", 0.0))
        r = (-d / 0.01 - th / 10 - 0.5 * np.log10(d * 1000 + 0.1) - 0.5 * np.log10(th + 0.1)
             + ((d < POS_TOL) & (th < ROT_TOL_DEG)) - pen)
        return r.astype(np.float32) if np.ndim(r) else float(r)

    def step(self, action):
        dq = np.clip(np.asarray(action, float), -1, 1) * MAX_DQ * self.scale
        self.q = np.clip(self.q + dq, self.lo, self.hi)
        self.steps += 1
        obs = self._obs()
        d, th = (float(v) for v in goal_errors(self.achieved, self.desired))
        table = touches_table(self.model, self.data)
        pinned = int(np.sum((self.q <= self.lo + 1e-6) | (self.q >= self.hi - 1e-6)))
        penalty = self.limit_penalty * pinned + (self.table_penalty if table else 0.0)
        info = {"success": d < POS_TOL and th < ROT_TOL_DEG and not table, "pos_err": d, "rot_err_deg": th,
                "table": table, "penalty": penalty, "steps": self.steps}
        r = self.compute_reward(self.achieved, self.desired, info)
        return obs, r, False, self.steps >= self.max_steps, info
