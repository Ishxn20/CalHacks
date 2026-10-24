"""Reach policy vs Kinematics.ik_topdown on held-out top-down targets.

    python -m sim.eval_reach [--policy sim/policy/reach_policy.zip] [--n 500]
    python -m sim.eval_reach --visual --residual sim/policy/reach_R1_best.zip   (camera-guided acceptance run:
        residual / IBVS / blind B / ik_topdown, plus a nudged-target run, on the same 200 held-out episodes)

Targets and start poses come from ReachEnv's sampler with seed 12345 (training: seeds 1-9, model selection:
seed 777), so every target
is one ik_topdown can reach. Both solvers get the same start (= IK seed); allow_flip=False for both.
Errors are measured with pinocchio FK. "Steps": policy actions until within 3 mm / 3 deg; for IK, damped
least-squares iterations (Jacobian evaluations) over all its seeds and yaw candidates.
"""
from __future__ import annotations

import argparse
import json
import sys
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


SCAN_SIGMA_M, SCAN_SIGMA_YAW = 0.0017, np.radians(3)   # 3 px / 3 deg on the 0.57 mm/px top-down scan


def _true_errors(kin, q, goal) -> tuple[float, float]:
    """Tool vs goal: mm, deg (jaws either way round)."""
    p, R = kin.fk(q)
    th = min(np.degrees(np.arccos(np.clip((np.trace(kin._target_rotation(y).T @ R) - 1) / 2, -1, 1)))
             for y in (goal[3], goal[3] + np.pi))
    return float(np.linalg.norm(p - np.array(goal[:3])) * 1000), float(th)


def _stats(rows: list[dict]) -> dict:
    pe, ro = np.array([r["pos_mm"] for r in rows]), np.array([r["rot_deg"] for r in rows])
    return {"n": len(rows), "success": float(np.mean([r["success"] for r in rows])),
            "pos_mm_median": float(np.median(pe)), "pos_mm_p90": float(np.percentile(pe, 90)),
            "rot_deg_median": float(np.median(ro)), "rot_deg_p90": float(np.percentile(ro, 90))}


def open_loop(kind: str, blind: ReachPolicy | None, n: int, seed: int, nudge: bool, mount: float) -> list[dict]:
    """ik_topdown / blind policy on the camera eval's episodes, aimed once at the scan's (noisy) estimate."""
    from sim.servo_env import VisualReachEnv, make_target

    env = VisualReachEnv(seed=seed, nudge=nudge, mount_noise=mount)
    rng, noise = np.random.default_rng(seed), np.random.default_rng(seed + 99)
    targets, rows = [make_target(rng) for _ in range(n)], []
    for i, t in enumerate(targets):
        env.reset(seed=seed + 1 + i, options={"target": t})
        x, y, z, yaw = env.goal
        est = (x + noise.normal(0, SCAN_SIGMA_M), y + noise.normal(0, SCAN_SIGMA_M), z,
               yaw + noise.normal(0, SCAN_SIGMA_YAW))
        goal = env.goal
        if nudge:                                     # it moves after the arm has committed; open loop can't see it
            dxy, rot = env.push
            goal = (x + dxy[0], y + dxy[1], z, yaw + rot)
        if kind == "ik":
            q = env.kin.ik_topdown(*est, env.q, (env.lo, env.hi), allow_flip=True)
        else:
            sols = [blind.rollout(*est[:3], yy, env.q) for yy in (est[3], est[3] + np.pi)]
            q = next((s[0] for s in sols if s[0] is not None), None)
            q = blind.env.q if q is None else q
        pe, ro = _true_errors(env.kin, q, goal) if q is not None else (np.inf, np.inf)
        rows.append({"fid": t["fid"], "success": bool(pe <= POS_TOL * 1000 and ro <= ROT_TOL_DEG),
                     "pos_mm": pe, "rot_deg": ro})
    return rows


def visual(argv) -> None:
    """Camera-guided reach acceptance run: one fixed set of held-out episodes, every method."""
    from stable_baselines3 import SAC

    from sim.ibvs import IBVS, ResidualEnv
    from sim.servo_env import evaluate

    ap = argparse.ArgumentParser(prog="sim.eval_reach --visual")
    ap.add_argument("--visual", action="store_true")
    ap.add_argument("--residual", required=True, help="residual SAC .zip (sim.train_reach --env residual)")
    ap.add_argument("--blind", default="sim/policy/reach_B_best.zip")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--mount-noise", type=float, default=1.0)
    ap.add_argument("--out", default="sim/policy/reach_visual_eval_200.json")
    args = ap.parse_args(argv)
    res_model = SAC.load(args.residual, env=ResidualEnv(pool_size=1), device="cpu")
    blind = ReachPolicy(args.blind)
    rows, out = {}, {"n": args.n, "seed": args.seed, "mount_noise": args.mount_noise, "residual": args.residual,
                     "blind": args.blind, "noise": "3 px, 3 deg, 7 % dropped frames, finger occlusion",
                     "success": "true tool pose within 3 mm and 3 deg at the last of 150 steps"}
    for nudge in (False, True):
        tag = "_nudged" if nudge else ""
        for name, model, cls in (("residual", res_model, ResidualEnv), ("ibvs", IBVS(), None)):
            rows[name + tag] = []
            evaluate(model, args.n, args.seed, env_cls=cls, rows=rows[name + tag], nudge=nudge,
                     mount_noise=args.mount_noise)
            print(name + tag, json.dumps(_stats(rows[name + tag])), flush=True)
        for name in ("blind", "ik"):
            rows[name + tag] = open_loop(name, blind, args.n, args.seed, nudge, args.mount_noise)
            print(name + tag, json.dumps(_stats(rows[name + tag])), flush=True)
    out["summary"] = {k: _stats(v) for k, v in rows.items()}
    out["episodes"] = rows
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["summary"], indent=1))


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if "--visual" in argv:
        return visual(argv)
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
