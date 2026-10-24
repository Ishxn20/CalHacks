"""Train the reach-only policy (sim/reach_env.py) with stable-baselines3 SAC on CPU.

    python -m sim.train_reach --name A                       # adaptive step scale + steep reward
    python -m sim.train_reach --name B --her                 # + HER (future, 4 goals)
    python -m sim.train_reach --name C --her --gamma 0.99

Every --eval-every steps the policy is rolled out DETERMINISTICALLY on 200 fixed held-out targets
(seed 777, scan-pose start half the time) and scored with pinocchio FK; the curve goes to
sim/policy/reach_<name>_curve.csv, the latest weights to reach_<name>.zip, the best (success, then median
position error) to reach_<name>_best.zip.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "arm")]

import torch  # noqa: E402
from stable_baselines3 import SAC, HerReplayBuffer  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402

from sim.eval_reach import held_out, rl_rows, summary  # noqa: E402
from sim.reach_env import ReachEnv  # noqa: E402
from sim.reach_policy import ReachPolicy  # noqa: E402

OUT = Path(__file__).resolve().parent / "policy"
FIELDS = ["steps", "success", "pos_mm_median", "pos_mm_p95", "rot_deg_median", "steps_median_success"]


class HeldOut(BaseCallback):
    def __init__(self, name: str, every: int) -> None:
        super().__init__()
        self.name, self.every, self.best = name, every, None
        self.pol = ReachPolicy(None)
        self.cases = held_out(self.pol.env, 200, 777)
        self.fh = open(OUT / f"reach_{name}_curve.csv", "w", newline="")
        self.w = csv.DictWriter(self.fh, FIELDS)
        self.w.writeheader()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every:
            return True
        self.pol.model = self.model
        s = summary(rl_rows(self.pol, self.pol.env.kin, self.cases))
        row = {"steps": self.num_timesteps, **{k: s[k] for k in FIELDS[1:]}}
        self.w.writerow(row)
        self.fh.flush()
        print(json.dumps(row), flush=True)
        self.model.save(OUT / f"reach_{self.name}.zip")
        key = (s["success"], -s["pos_mm_median"])
        if self.best is None or key > self.best:
            self.best = key
            self.model.save(OUT / f"reach_{self.name}_best.zip")
        return True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sim.train_reach")
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--her", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-every", type=int, default=50_000)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)            # fastest for a 256x256 MLP, and leaves cores for the YOLO job
    OUT.mkdir(exist_ok=True)
    her = {"replay_buffer_class": HerReplayBuffer,
           "replay_buffer_kwargs": {"n_sampled_goal": 4, "goal_selection_strategy": "future"}} if args.her else {}
    model = SAC("MultiInputPolicy", ReachEnv(seed=args.seed), policy_kwargs={"net_arch": [256, 256]},
                learning_rate=3e-4, buffer_size=1_000_000, batch_size=256, gamma=args.gamma, learning_starts=5000,
                device="cpu", seed=args.seed, verbose=0, **her)
    model.learn(args.steps, callback=HeldOut(args.name, args.eval_every))


if __name__ == "__main__":
    main()
