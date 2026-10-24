"""Stress-test the declutter planner on physics-settled piles.

Each episode: settle a pile in MuJoCo, render it, give the planner the exact instance masks (segmentation
quality is measured separately by training), and execute its plan: a moved instrument leaves the pile
(placed in staging), the pile is re-settled, repeat until the target is grasped or the planner gives up.

    python -m sim.declutter_eval --episodes 200          (from the repo root)
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))

from scrub_vision.identify import identify  # noqa: E402
from scrub_vision.planner import plan  # noqa: E402
from scrub_vision.render import CamSpec, render_scene, scene_mesh  # noqa: E402
from scrub_vision.templates import build_templates, load_instruments  # noqa: E402

from sim.pile_sim import settle_pile  # noqa: E402

CAM = CamSpec()
S = CAM.height_mm / CAM.fx
H = np.array([[S, 0, -CAM.w / 2 * S], [0, S, -CAM.h / 2 * S], [0, 0, 1]])
STAGING = [(400.0, 400.0 + 60 * i) for i in range(6)]     # off-camera: staged items never re-enter the view
_cache: dict = {}


def _assets():
    if not _cache:
        _cache["insts"] = load_instruments()
        _cache["templates"] = build_templates(_cache["insts"], 0.5)
    return _cache["insts"], _cache["templates"]


def run_episode(rng: np.random.Generator, n_items: int | None = None, items: list[str] | None = None,
                target: str | None = None, max_moves: int = 6) -> dict:
    insts, templates = _assets()
    if items is None:
        n = n_items or int(rng.integers(3, 9))
        items = [str(f) for f in rng.choice(list(insts), size=n, replace=False)]
    target = target or str(rng.choice(items))
    remaining, used, moves = list(items), set(), 0
    while True:
        pile, _ = settle_pile(remaining, rng)
        _, masks = render_scene([(scene_mesh(insts[f]), T) for f, T in pile], CAM, rng)
        dets, truth = [], {}
        for (fid, _), m in zip(pile, masks):
            if m.sum() < 300:
                continue
            d = identify(m, None, H, templates, insts)
            dets.append(d)
            truth[id(d)] = fid
        p = plan(dets, target, H, STAGING, used)
        if p["action"] == "fail":
            return {"outcome": "fail", "moves": moves, "say": p["say"], "target": target, "items": items}
        chosen = min(dets, key=lambda d: np.hypot(d.grasp_mm[0] - p["target"]["x_mm"],
                                                  d.grasp_mm[1] - p["target"]["y_mm"]))
        actual = truth[id(chosen)]
        if p["action"] == "grasp_target":
            ok = actual == target or actual in (insts[target].same_look_as,)
            outcome = "grasped" if ok else ("ambiguous" if p["target"]["ambiguous"] else "wrong")
            return {"outcome": outcome, "moves": moves, "grasped_id": actual,
                    "say": p["say"], "target": target, "items": items, "ambiguous": p["target"]["ambiguous"]}
        if actual == target:
            return {"outcome": "wrong", "moves": moves, "grasped_id": actual, "say": "moved the target aside",
                    "target": target, "items": items}
        remaining.remove(actual)
        used.add(p["staging_index"])
        moves += 1
        if moves >= max_moves:
            return {"outcome": "cap", "moves": moves, "say": "", "target": target, "items": items}


def summarise(results: list[dict]) -> dict:
    n = len(results)
    ok = [r for r in results if r["outcome"] == "grasped"]
    return {"n": n, "success_rate": len(ok) / n if n else 0.0,
            "outcomes": dict(collections.Counter(r["outcome"] for r in results)),
            "mean_moves_success": float(np.mean([r["moves"] for r in ok])) if ok else 0.0,
            "fail_reasons": dict(collections.Counter(r["say"].split(" the ")[0] for r in results
                                                     if r["outcome"] == "fail")),
            "wrong_targets": dict(collections.Counter(f"{r['target']}->{r.get('grasped_id')}" for r in results
                                                      if r["outcome"] == "wrong"))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sim.declutter_eval")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sim/declutter_eval.json")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    results = []
    for i in range(args.episodes):
        results.append(run_episode(rng))
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{args.episodes}: {summarise(results)['success_rate']:.0%} success", flush=True)
    s = summarise(results)
    Path(args.out).write_text(json.dumps({"summary": s, "episodes": results}, indent=1, default=str))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
