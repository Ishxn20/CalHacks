"""Reach-only RL environment: drive the B601-RS tool frame to a top-down pose (x, y, z, jaw yaw).

Kinematic MuJoCo rollout (the real servos track position setpoints, so a step = a new setpoint):
action = joint-position deltas in [-1, 1] * MAX_DQ rad, clipped to arm.yaml limits.
Observation: joint angles, target pose, current tool pose, position and orientation error.
Reward: -(pos error / 1 cm) - (rot error / 10 deg), +SUCCESS_BONUS and done inside 3 mm and 3 deg,
-limit_penalty per joint pinned at a limit, -table_penalty if any arm geom touches the table.
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

MAX_DQ = 0.04          # rad per step
POS_TOL, ROT_TOL_DEG = 3e-3, 3.0
SUCCESS_BONUS = 10.0
SCAN = np.radians([0.0, 45.0, 70.0, 0.0, 65.0, 0.0])   # arm/config/poses.yaml scan


def rot_err(R: np.ndarray, R_target: np.ndarray) -> np.ndarray:
    """Rotation vector (world frame) taking R to R_target."""
    return _log3(R_target @ R.T)


def _log3(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-6:
        return v / 2
    if np.pi - th < 1e-4:                      # ~180 deg: axis from the symmetric part
        w, V = np.linalg.eigh((R + np.eye(3)) / 2)
        return V[:, -1] * th
    return v * th / (2 * np.sin(th))


class ReachEnv(gym.Env):
    def __init__(self, seed: int | None = None, max_steps: int = 100, table_penalty: float = 5.0,
                 limit_penalty: float = 0.1, precision: float = 0.0) -> None:
        cfg = load_arm_config()
        self.lo, self.hi, self.table_z = cfg.lower, cfg.upper, cfg.pick.table_z
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.kin = Kinematics()
        self.max_steps, self.table_penalty, self.limit_penalty = max_steps, table_penalty, limit_penalty
        self.precision = precision        # weight of a bounded bonus that keeps paying for the last few mm/deg
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (29,), np.float32)
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

    def set_target(self, target, q0) -> np.ndarray:
        self.target = tuple(float(v) for v in target)
        self.R_target = self.kin._target_rotation(self.target[3])
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
    def _errors(self):
        set_arm(self.model, self.data, self.q)
        p, R = tool_pose(self.model, self.data)
        e_p = np.array(self.target[:3]) - p
        e_r = rot_err(R, self.R_target)
        return p, R, e_p, e_r

    def _obs(self) -> np.ndarray:
        p, R, e_p, e_r = self._errors()
        x, y, z, yaw = self.target
        return np.concatenate([self.q / np.pi, np.array([x, y, z]) / 0.4, [np.sin(yaw), np.cos(yaw)], p / 0.4,
                               R[:, :2].T.ravel(), e_p * 10, np.tanh(e_p * 200), e_r]).astype(np.float32)

    def step(self, action):
        dq = np.clip(np.asarray(action, float), -1, 1) * MAX_DQ
        self.q = np.clip(self.q + dq, self.lo, self.hi)
        self.steps += 1
        obs = self._obs()
        _, _, e_p, e_r = self._errors()
        d, th = float(np.linalg.norm(e_p)), float(np.degrees(np.linalg.norm(e_r)))
        table = touches_table(self.model, self.data)
        pinned = int(np.sum((self.q <= self.lo + 1e-6) | (self.q >= self.hi - 1e-6)))
        success = d < POS_TOL and th < ROT_TOL_DEG and not table
        r = -d / 0.01 - th / 10 - self.limit_penalty * pinned - (self.table_penalty if table else 0.0)
        r += self.precision * (2 - np.tanh(d / 0.005) - np.tanh(th / 5))
        if success:
            r += SUCCESS_BONUS
        truncated = self.steps >= self.max_steps
        return obs, float(r), success, truncated, {"success": success, "pos_err": d, "rot_err_deg": th,
                                                    "table": table, "steps": self.steps}
