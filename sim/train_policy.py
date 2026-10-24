"""Train the grasp-selection policy (sim/rl_policy.py) with REINFORCE + mean-return baseline on CPU.

    python -m sim.train_policy --iters 150 --workers 6          (from the repo root)

Each iteration: every worker plays --episodes-per-worker sampled episodes in DeclutterEnv with the current
weights; returns-to-go minus the batch mean weight the log-prob gradient of a linear softmax,
grad log pi(a) = phi_a - E_pi[phi]. Every --val-every iterations the greedy policy is scored on a fixed
validation set (seed 999, never the seed-0 test set) and the best one is saved; the latest weights
are always kept as <out>_last.json.
"""
from __future__ import annotations

import argparse
import csv
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))

from sim.rl_env import DeclutterEnv  # noqa: E402
from sim.rl_policy import N_FEATURES, Policy  # noqa: E402

OUT = Path(__file__).resolve().parent / "policy"


def rollouts(args) -> list[dict]:
    w, seed, n, greedy = args
    pol, rng = Policy(w), np.random.default_rng(seed)
    env, out = DeclutterEnv(rng), []
    for _ in range(n):
        obs, steps, done = env.reset(), [], False
        while not done:
            a = pol.act(obs["features"], None if greedy else rng)
            feats = obs["features"]
            obs, r, done, info = env.step(a)
            steps.append((feats, a, r))
        out.append({"steps": steps, "outcome": info["outcome"], "moves": info["moves"]})
    return out


def gradient(w: np.ndarray, episodes: list[dict]) -> np.ndarray:
    pol, items = Policy(w), []
    for ep in episodes:
        rs = [r for _, _, r in ep["steps"]]
        for t, (feats, a, _) in enumerate(ep["steps"]):
            items.append((feats, a, sum(rs[t:])))
    b = np.mean([g for *_, g in items])
    grad = np.zeros_like(w)
    for feats, a, g in items:
        grad += (g - b) * (feats[a] - pol.probs(feats) @ feats)
    return grad / len(episodes)


def play(pool: Pool, w: np.ndarray, seed: int, n: int, workers: int, greedy: bool) -> list[dict]:
    per = -(-n // workers)
    return [e for chunk in pool.map(rollouts, [(w, seed + k, per, greedy) for k in range(workers)]) for e in chunk]


def rate(eps: list[dict], outcome: str = "grasped") -> float:
    return float(np.mean([e["outcome"] == outcome for e in eps]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sim.train_policy")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--episodes-per-worker", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--val-every", type=int, default=25)
    ap.add_argument("--val-episodes", type=int, default=120)
    ap.add_argument("--out", default=str(OUT / "grasp_policy.json"))
    args = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    w, m, v = np.zeros(N_FEATURES), np.zeros(N_FEATURES), np.zeros(N_FEATURES)   # Adam state
    best, log = -np.inf, []
    with Pool(args.workers) as pool, open(OUT / "training_curve.csv", "w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(["iter", "episodes", "train_success", "train_wrong", "val_success", "val_wrong"])
        for it in range(1, args.iters + 1):
            eps = play(pool, w, 10_000 + it * args.workers, args.episodes_per_worker * args.workers,
                       args.workers, greedy=False)
            g = gradient(w, eps)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            w = w + args.lr * 0.1 * (m / (1 - 0.9 ** it)) / (np.sqrt(v / (1 - 0.999 ** it)) + 1e-8)
            row = [it, it * len(eps), rate(eps), rate(eps, "wrong"), "", ""]
            if it % args.val_every == 0 or it == args.iters:
                val = play(pool, w, 999_000, args.val_episodes, args.workers, greedy=True)
                row[4:] = [rate(val), rate(val, "wrong")]
                score = rate(val) - rate(val, "wrong")
                if score > best:
                    best = score
                    Policy(w).save(args.out, iter=it, val_success=rate(val), val_wrong=rate(val, "wrong"))
            Policy(w).save(Path(args.out).with_name(Path(args.out).stem + "_last.json"), iter=it)
            cw.writerow(row)
            fh.flush()
            log.append(row)
            print(*row, flush=True)
    plot(log)


def plot(log: list[list]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    it = np.array([r[0] for r in log])
    tr = np.array([r[2] for r in log])
    k = min(10, len(tr))
    plt.figure(figsize=(7, 4))
    plt.plot(it, tr, alpha=0.3, label="train success (sampled)")
    plt.plot(it[k - 1:], np.convolve(tr, np.ones(k) / k, "valid"), label=f"train, {k}-iter mean")
    vi = [(r[0], r[4]) for r in log if r[4] != ""]
    if vi:
        plt.plot(*zip(*vi), "o-", label="validation success (greedy)")
    plt.xlabel("iteration")
    plt.ylabel("target grasped")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "training_curve.png", dpi=120)


if __name__ == "__main__":
    main()
