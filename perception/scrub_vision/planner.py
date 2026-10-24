"""Rule-based declutter planner: grasp the requested instrument if it's on top, else move the best
exposed instrument (preferably one lying on the target) to the next free staging spot.

Output is the /plan JSON the arm consumes (arm/scrub_arm/perception_client.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from .identify import Detection, above, feasible, on_top

TRAY_YAML = Path(__file__).resolve().parents[2] / "config" / "tray.yaml"
STAGED_RADIUS_MM = 60.0
_names: dict[str, str] | None = None


def display_name(fid: str) -> str:
    global _names
    if _names is None:
        try:
            _names = {k: v.get("name", k) for k, v in yaml.safe_load(TRAY_YAML.read_text())["instruments"].items()}
        except (OSError, KeyError, TypeError):
            _names = {}
    return _names.get(fid, fid.replace("_", " "))


def matches(d: Detection, target_id: str) -> bool:
    return d.fine_id == target_id or target_id in d.alt_ids or (d.fine_id is None and target_id in d.candidates)


def _target_json(d: Detection, fid: str | None, ambiguous: bool = False) -> dict:
    return {"id": fid, "x_mm": round(d.grasp_mm[0], 1), "y_mm": round(d.grasp_mm[1], 1),
            "yaw_deg": round(d.yaw_deg, 1), "thickness_mm": d.thickness_mm,
            "confidence": round(d.confidence, 3), "exposure": round(d.exposure, 3), "ambiguous": ambiguous}


def plan(dets: list[Detection], target_id: str, H: np.ndarray, staging: list[tuple[float, float]],
         used_staging: set[int], staged_xy: list[tuple[float, float]] | None = None,
         exposure_min: float = 0.7, conf_min: float = 0.5, staging_yaw_deg: float = 0.0) -> dict:
    name = display_name(target_id)
    staged_xy = staged_xy or []

    def in_staging(d: Detection) -> bool:
        return any(np.hypot(d.grasp_mm[0] - x, d.grasp_mm[1] - y) < STAGED_RADIUS_MM for x, y in staged_xy)

    dets = [d for d in dets if d.candidates]          # blobs no instrument template explains: not instruments
    grabbable = [d for d in dets if on_top(d, dets) and feasible(d, dets, H)]
    targets = sorted((d for d in dets if matches(d, target_id)), key=lambda d: -d.confidence)
    for t in targets:
        if t in grabbable and t.exposure >= exposure_min and t.confidence >= conf_min:
            ambiguous = t.fine_id != target_id and target_id not in t.alt_ids
            say = f"I think this is the {name}" if ambiguous else name
            return {"action": "grasp_target", "target": _target_json(t, target_id, ambiguous), "place": None,
                    "say": say}

    movers = [d for d in grabbable if not matches(d, target_id) and not in_staging(d)]
    if not movers:
        why = f"I can't get to the {name}" if targets else f"I don't see a {name}"
        return {"action": "fail", "target": None, "place": None, "say": why}
    free = [i for i in range(len(staging)) if i not in used_staging]
    if not free:
        return {"action": "fail", "target": None, "place": None,
                "say": f"The staging area is full and the {name} is still buried"}

    def priority(d: Detection) -> tuple:
        on_target = any(above(d.mask, t.mask) for t in targets)
        return (not on_target, -d.confidence)

    m = sorted(movers, key=priority)[0]
    x, y = staging[free[0]]
    return {"action": "move_aside", "target": _target_json(m, m.fine_id), "staging_index": free[0],
            "place": {"x_mm": x, "y_mm": y, "yaw_deg": staging_yaw_deg},
            "say": f"Moving the {display_name(m.fine_id) if m.fine_id else m.coarse or 'instrument'} aside"}
