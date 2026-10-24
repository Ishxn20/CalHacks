"""The trained reach policy as an IK replacement: roll it out in the MuJoCo arm model, return the joint vector.

    ReachPolicy().solve(x, y, z, yaw, q_seed) -> joint vector (rad) or None if it didn't get within 3 mm / 3 deg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "arm"))

from sim.reach_env import ReachEnv  # noqa: E402

DEFAULT = Path(__file__).resolve().parent / "policy" / "reach_policy.zip"


class ReachPolicy:
    def __init__(self, path=DEFAULT, max_steps: int = 150) -> None:
        """path: a saved SAC .zip, or a live SAC model (evaluation during training)."""
        self.env = ReachEnv(max_steps=max_steps)
        if isinstance(path, (str, Path)):
            from stable_baselines3 import SAC

            path = SAC.load(str(path), env=self.env, device="cpu")   # HER models need the env to load
        self.model = path

    def rollout(self, x: float, y: float, z: float, yaw: float, q_seed) -> tuple[np.ndarray | None, int, dict]:
        """(q at first success or None, steps taken, last step info)."""
        obs = self.env.set_target((x, y, z, yaw), q_seed)
        info, self.path = {}, []                     # joint trajectory of the last rollout (for the video)
        for _ in range(self.env.max_steps):
            a, _ = self.model.predict(obs, deterministic=True)
            obs, _, _, truncated, info = self.env.step(a)
            self.path.append(self.env.q.copy())
            if info["success"]:
                return self.env.q.copy(), info["steps"], info
            if truncated:
                break
        return None, self.env.steps, info

    def solve(self, x: float, y: float, z: float, yaw: float, q_seed, allow_flip: bool = True) -> np.ndarray | None:
        """Like Kinematics.ik_topdown: allow_flip also tries yaw + pi and keeps the answer nearest q_seed."""
        q_seed = np.asarray(q_seed, float)
        sols = [q for yy in ((yaw, yaw + np.pi) if allow_flip else (yaw,))
                if (q := self.rollout(x, y, z, (yy + np.pi) % (2 * np.pi) - np.pi, q_seed)[0]) is not None]
        return min(sols, key=lambda q: np.linalg.norm(q - q_seed)) if sols else None
