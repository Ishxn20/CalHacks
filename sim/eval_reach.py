"""Reach policy vs Kinematics.ik_topdown on held-out top-down targets.

    python -m sim.eval_reach [--policy sim/policy/reach_policy.zip] [--n 500]

Targets and start poses come from ReachEnv's sampler with seed 12345 (training: seeds 1-9, model selection:
seed 777), so every target
is one ik_topdown can reach. Both solvers get the same start (= IK seed); allow_flip=False for both.
Errors are measured with pinocchio FK. "Steps": policy actions until within 3 mm / 3 deg; for IK, damped
least-squares iterations (Jacobian evaluations) over all its seeds and yaw candidates.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from sim.reach_env import POS_TOL, ROT_TOL_DEG, SCAN, ReachEnv
from sim.reach_policy import DEFAULT, ReachPolicy


class _Counting:
    """pinocchio proxy that counts Jacobian evaluations (= DLS iterations)."""

    def __init__(self, pin) -> None:
        self._pin, self.n = pin, 0

    def __getattr__(self, k):
        return getattr(self._pin, k)

    def computeFrameJacobian(self, *a):
        self.n += 1
        return self._pin.computeFrameJacobian(*a)


def errors(kin, q, target) -> tuple[float, float]:
    p, R = kin.fk(q)
    Rt = kin._target_rotation(target[3])
    return (float(np.linalg.norm(p - np.array(target[:3]))) * 1000,
            float(np.degrees(np.arccos(np.clip((np.trace(Rt.T @ R) - 1) / 2, -1, 1)))))


def held_out(env: ReachEnv, n: int, seed: int) -> list[tuple]:
    """n (target, start q) pairs: start at the scan pose half the time, else at another target's IK solution."""
    rng, cases = np.random.default_rng(seed), []
    for _ in range(n):
        q0 = env.sample_target(rng)[1] if rng.random() >= 0.5 else SCAN
        cases.append((env.sample_target(rng)[0], q0))
    return cases


def rl_rows(pol: ReachPolicy, kin, cases) -> list[dict]:
    """Deterministic rollouts; errors by pinocchio FK at the first in-tolerance step, else the last step."""
    rows = []
    for target, q0 in cases:
        t = time.perf_counter()
        q, steps, _ = pol.rollout(*target, q0)
        ms = (time.perf_counter() - t) * 1000
        pe, ro = errors(kin, q if q is not None else pol.env.q, target)
        rows.append({"pos_mm": pe, "rot_deg": ro, "ok": q is not None and pe <= POS_TOL * 1000
                     and ro <= ROT_TOL_DEG, "steps": steps, "ms": ms})
    return rows


def summary(rows: list[dict]) -> dict:
    pe = np.array([r["pos_mm"] for r in rows])
    re_ = np.array([r["rot_deg"] for r in rows])
    ok = np.array([r["ok"] for r in rows])
    st = np.array([r["steps"] for r in rows])
    return {"n": len(rows), "success": float(ok.mean()), "pos_mm_mean": float(pe.mean()),
            "pos_mm_median": float(np.median(pe)), "pos_mm_p95": float(np.percentile(pe, 95)),
            "rot_deg_mean": float(re_.mean()), "rot_deg_median": float(np.median(re_)),
            "steps_median_success": float(np.median(st[ok])) if ok.any() else None,
            "steps_mean_success": float(st[ok].mean()) if ok.any() else None,
            "ms_per_solve": float(np.mean([r["ms"] for r in rows]))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sim.eval_reach")
    ap.add_argument("--policy", default=str(DEFAULT))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--out", default="sim/policy/reach_eval.json")
    args = ap.parse_args(argv)
    env, pol = ReachEnv(), ReachPolicy(args.policy)
    kin = env.kin
    cases = held_out(env, args.n, 12345)
    counter = _Counting(kin.pin)
    kin.pin = counter
    res = {"ik": [], "rl": []}
    for target, q0 in cases:
        counter.n, t = 0, time.perf_counter()
        q = kin.ik_topdown(*target, q0, (env.lo, env.hi), allow_flip=False)
        ms = (time.perf_counter() - t) * 1000
        pe, ro = errors(kin, q, target) if q is not None else (np.inf, np.inf)
        res["ik"].append({"pos_mm": pe, "rot_deg": ro, "ok": pe <= POS_TOL * 1000 and ro <= ROT_TOL_DEG,
                          "steps": counter.n, "ms": ms})
    res["rl"] = rl_rows(pol, kin, cases)
    out = {k: summary(v) for k, v in res.items()}
    Path(args.out).write_text(json.dumps({"summary": out, "policy": args.policy, "targets": [
        list(t) for t, _ in cases], **res}, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
