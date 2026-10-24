"""Learned grasp selection (spec §5.6): a linear scorer over per-detection features, softmax over the legal
choices plus a "stop" choice. Safety stays hard-coded: only recognised, on-top, feasible detections are
choices at all; the policy only learns their order. Arm motion is not learned.

    pol = Policy.load("sim/policy/grasp_policy.json")
    pol.choose(dets, target_id, H, staging, used_staging)   -> same JSON as planner.plan()
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scrub_vision.identify import Detection, above, feasible, on_top
from scrub_vision.planner import (SURE_EXPOSURE, STAGED_RADIUS_MM, _target_json, display_name, matches,
                                  might_be)
from scrub_vision.templates import classes, load_instruments

COARSE = classes()
NAMES = ["is_det", "stop", "matches", "sure", "ambiguous", "might_be", "same_coarse", "exposure", "confidence",
         "above_target", "above_might_be", "n_covered", "dist_centre", "stop_target_seen"] + \
        [f"coarse_{c}" for c in COARSE]
N_FEATURES = len(NAMES)
_coarse_of: dict[str, str] = {}


def legal(dets: list[Detection], H: np.ndarray, staged_xy=()) -> list[Detection]:
    """The only detections the policy may pick: recognised, nothing on top, jaws fit. Same filter as plan()."""
    return [d for d in dets if d.candidates and on_top(d, dets) and feasible(d, dets, H)
            and not any(np.hypot(d.grasp_mm[0] - x, d.grasp_mm[1] - y) < STAGED_RADIUS_MM for x, y in staged_xy)]


def features(choices: list[Detection], dets: list[Detection], target_id: str) -> np.ndarray:
    """One row per choice, then the stop row."""
    if not _coarse_of:
        _coarse_of.update({k: v.coarse for k, v in load_instruments().items()})
    targets = [d for d in dets if matches(d, target_id)]
    suspects = [d for d in dets if might_be(d, target_id)]
    centre = np.mean([d.grasp_mm for d in dets], axis=0) if dets else np.zeros(2)
    rows = []
    for d in choices:
        f = dict.fromkeys(NAMES, 0.0)
        m = matches(d, target_id)
        f.update(is_det=1, matches=m, exposure=d.exposure, confidence=d.confidence,
                 sure=m and d.exposure >= SURE_EXPOSURE and d.confidence >= 0.5,
                 ambiguous=m and d.fine_id != target_id and target_id not in d.alt_ids,
                 might_be=not m and might_be(d, target_id), same_coarse=d.coarse == _coarse_of.get(target_id),
                 above_target=any(above(d.mask, t.mask) for t in targets if t is not d),
                 above_might_be=any(above(d.mask, t.mask) for t in suspects if t is not d),
                 n_covered=sum(above(d.mask, o.mask) for o in dets if o is not d) / 4,
                 dist_centre=float(np.hypot(*(np.array(d.grasp_mm) - centre))) / 100)
        if d.coarse in COARSE:
            f[f"coarse_{d.coarse}"] = 1
        rows.append([float(f[k]) for k in NAMES])
    stop = dict.fromkeys(NAMES, 0.0)
    stop.update(stop=1, stop_target_seen=float(bool(targets)))
    rows.append([stop[k] for k in NAMES])
    return np.array(rows)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


class Policy:
    def __init__(self, w: np.ndarray) -> None:
        self.w = np.asarray(w, float)

    @classmethod
    def load(cls, path) -> "Policy":
        d = json.loads(Path(path).read_text())
        assert d["features"] == NAMES, "policy was trained on a different feature set"
        return cls(np.array(d["w"]))

    def save(self, path, **meta) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"features": NAMES, "w": self.w.tolist(), **meta}, indent=1))

    def probs(self, feats: np.ndarray) -> np.ndarray:
        return softmax(feats @ self.w)

    def act(self, feats: np.ndarray, rng: np.random.Generator | None = None) -> int:
        """Index into the choices (len(choices) = stop). Greedy unless an rng is given (training)."""
        p = self.probs(feats)
        return int(np.argmax(p)) if rng is None else int(rng.choice(len(p), p=p))

    def choose(self, dets: list[Detection], target_id: str, H: np.ndarray, staging: list[tuple[float, float]],
               used_staging: set[int], staged_xy=None, staging_yaw_deg: float = 0.0, **_) -> dict:
        name = display_name(target_id)
        choices = legal(dets, H, staged_xy or ())
        a = self.act(features(choices, dets, target_id))
        seen = any(matches(d, target_id) for d in dets)
        if a == len(choices):
            why = f"I can't get to the {name}" if seen else f"I don't see a {name}"
            return {"action": "fail", "target": None, "place": None, "say": why}
        d = choices[a]
        if matches(d, target_id):
            ambiguous = d.fine_id != target_id and target_id not in d.alt_ids
            return {"action": "grasp_target", "target": _target_json(d, target_id, ambiguous), "place": None,
                    "say": f"I think this is the {name}" if ambiguous else name}
        free = [i for i in range(len(staging)) if i not in used_staging]
        if not free:
            return {"action": "fail", "target": None, "place": None,
                    "say": f"The staging area is full and the {name} is still buried"}
        x, y = staging[free[0]]
        return {"action": "move_aside", "target": _target_json(d, d.fine_id), "staging_index": free[0],
                "place": {"x_mm": x, "y_mm": y, "yaw_deg": staging_yaw_deg},
                "say": f"Moving the {display_name(d.fine_id) if d.fine_id else d.coarse or 'instrument'} aside"}
