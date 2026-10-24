"""Train the reach-only policy (sim/reach_env.py) with stable-baselines3 SAC on CPU.

    python -m sim.train_reach --steps 600000            (from the repo root)

Saves --out (latest weights, every 10k steps) and a success-rate curve beside it (<out>_curve.csv).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "arm")]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from sim.reach_env import ReachEnv  # noqa: E402

OUT = Path(__file__).resolve().parent / "policy"


class Curve(BaseCallback):
    """Every `every` steps: success rate and mean final error of the episodes finished since, then save."""

    def __init__(self, path: Path, every: int, out: str) -> None:
        super().__init__()
        self.every, self.out, self.eps = every, out, []
        self.fh = open(path, "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(["steps", "episodes", "success", "final_pos_err_mm", "final_rot_err_deg"])

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.eps.append((info["success"], info["pos_err"], info["rot_err_deg"]))
        if self.num_timesteps % self.every == 0 and self.eps:
            s, p, r = map(np.array, zip(*self.eps))
            row = [self.num_timesteps, len(self.eps), s.mean(), np.median(p) * 1000, np.median(r)]
            self.w.writerow(row)
            self.fh.flush()
            print(*[round(float(v), 3) for v in row], flush=True)
            self.eps = []
            self.model.save(self.out)
        return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sim.train_reach")
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--threads", type=int, default=1)     # 1 is fastest for a 256x256 MLP
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--precision", type=float, default=0.0, help="ReachEnv precision-bonus weight")
    ap.add_argument("--out", default=str(OUT / "reach_policy.zip"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    torch.set_num_threads(args.threads)
    OUT.mkdir(exist_ok=True)
    model = SAC("MlpPolicy", ReachEnv(seed=args.seed, precision=args.precision), policy_kwargs={"net_arch": [256, 256]}, learning_rate=3e-4,
                buffer_size=1_000_000, batch_size=256, gamma=0.98, learning_starts=5000, device="cpu", seed=args.seed,
                verbose=0)
    model.learn(args.steps, callback=Curve(out.with_name(out.stem + "_curve.csv"), 10_000, args.out))
    model.save(args.out)


if __name__ == "__main__":
    main()
