"""Train the reach-only policy (sim/reach_env.py) with stable-baselines3 SAC on CPU.

    python -m sim.train_reach --name A                       # adaptive step scale + steep reward
    python -m sim.train_reach --name B --her                 # + HER (future, 4 goals)
    python -m sim.train_reach --name C --her --gamma 0.99
    python -m sim.train_reach --resume sim/policy/checkpoints/B8          # continue a paused run to its target

Every --eval-every steps the policy is rolled out DETERMINISTICALLY on 200 fixed held-out targets
(seed 777, scan-pose start half the time) and scored with pinocchio FK; the curve goes to
<out-dir>/reach_<name>_curve.csv, the latest weights to reach_<name>.zip, the best (success, then median
position error) to reach_<name>_best.zip, and a resumable checkpoint (model, replay buffer, meta.json) to
<out-dir>/checkpoints/<name>/. --stop-at N pauses the run after the evaluation at step N.
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
EVAL_SEED = 777


def _key(s: dict) -> tuple:
    return (float(s["success"]), -float(s["pos_mm_median"]))


class HeldOut(BaseCallback):
    def __init__(self, cfg: dict, out: Path, stop_at: int | None) -> None:
        super().__init__()
        self.cfg, self.out, self.stop_at = cfg, out, stop_at
        self.name, self.every = cfg["name"], cfg["eval_every"]
        self.stop, self.last = False, None
        self.pol = ReachPolicy(None)
        self.cases = held_out(self.pol.env, cfg["eval_n"], EVAL_SEED)
        curve = out / f"reach_{self.name}_curve.csv"
        rows = list(csv.DictReader(curve.open())) if curve.exists() and cfg.get("resumed") else []
        self.best = max(map(_key, rows), default=None)
        self.fh = open(curve, "a" if rows else "w", newline="")
        self.w = csv.DictWriter(self.fh, FIELDS)
        if not rows:
            self.w.writeheader()

    def _on_step(self) -> bool:
        return not self.stop                 # pause requested by the evaluation at --stop-at

    def _on_rollout_end(self) -> None:
        """After the step's transition is in the buffer (SB3 calls _on_step before storing it)."""
        if self.num_timesteps % self.every or self.num_timesteps == self.last:
            return
        self.last = self.num_timesteps
        self.pol.model = self.model
        s = summary(rl_rows(self.pol, self.pol.env.kin, self.cases))
        row = {"steps": self.num_timesteps, **{k: s[k] for k in FIELDS[1:]}}
        self.w.writerow(row)
        self.fh.flush()
        print(json.dumps(row), flush=True)
        self.model.save(self.out / f"reach_{self.name}.zip")
        if self.best is None or _key(s) > self.best:
            self.best = _key(s)
            self.model.save(self.out / f"reach_{self.name}_best.zip")
        checkpoint(self.model, self.out, self.cfg)
        self.stop = self.stop_at is not None and self.num_timesteps >= self.stop_at


def checkpoint(model: SAC, out: Path, cfg: dict) -> Path:
    d = out / "checkpoints" / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    model.save(d / "model.zip")
    model.save_replay_buffer(d / "replay_buffer.pkl")
    meta = {k: v for k, v in cfg.items() if k != "resumed"}
    (d / "meta.json").write_text(json.dumps({**meta, "num_timesteps": model.num_timesteps}, indent=1))
    return d


def main(argv=None) -> SAC:
    ap = argparse.ArgumentParser(prog="sim.train_reach")
    ap.add_argument("--name")
    ap.add_argument("--resume", help="checkpoint dir; config comes from its meta.json")
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--stop-at", type=int)
    ap.add_argument("--her", action="store_true")
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--learning-starts", type=int, default=5000)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args(argv)
    torch.set_num_threads(1)            # fastest for a 256x256 MLP, and leaves cores for the YOLO job
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        d = Path(args.resume)
        cfg = {**json.loads((d / "meta.json").read_text()), "resumed": True}
        model = SAC.load(d / "model.zip", env=ReachEnv(seed=cfg["seed"] + cfg["num_timesteps"]), device="cpu")
        if (d / "replay_buffer.pkl").exists():
            model.load_replay_buffer(d / "replay_buffer.pkl")
        else:
            print(f"no replay buffer in {d}: refilling one for {cfg['learning_starts']} steps first", flush=True)
            model.learning_starts = model.num_timesteps + cfg["learning_starts"]
    else:
        cfg = {"name": args.name, "her": args.her, "gamma": args.gamma, "seed": args.seed,
               "target_steps": args.steps, "eval_every": args.eval_every, "eval_n": args.eval_n,
               "eval_seed": EVAL_SEED, "learning_starts": args.learning_starts}
        her = {"replay_buffer_class": HerReplayBuffer,
               "replay_buffer_kwargs": {"n_sampled_goal": 4, "goal_selection_strategy": "future"}} if args.her else {}
        model = SAC("MultiInputPolicy", ReachEnv(seed=args.seed), policy_kwargs={"net_arch": [256, 256]},
                    learning_rate=3e-4, buffer_size=args.steps, batch_size=256, gamma=args.gamma,
                    learning_starts=args.learning_starts, device="cpu", seed=args.seed, verbose=0, **her)
    remaining = cfg["target_steps"] - model.num_timesteps
    if remaining > 0 and (args.stop_at is None or model.num_timesteps < args.stop_at):
        model.learn(remaining, callback=HeldOut(cfg, out, args.stop_at), reset_num_timesteps=not args.resume)
    return model


if __name__ == "__main__":
    main()
