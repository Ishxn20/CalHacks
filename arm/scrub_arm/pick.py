"""Turn one camera detection into joint targets for a top-down grip.

Pure maths, no hardware: the pick job (handoff.py) feeds it the vision service's answer and moves to what comes back.

    detection (pixels) -> undo camera drift -> grasp point -> checks -> reach map -> joints, wrist from grip angle
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import JointCfg, PickCfg
from .reach import ReachMap


class PickError(ValueError):
    """The detection can't be picked safely; the message is fit to read back to the user."""


@dataclass
class GraspPlan:
    name: str | None
    conf: float
    grasp_px: tuple[float, float]    # where the jaws go, in the calibrated (tray_view) image
    grip_deg: float
    width_mm: float | None
    anchor: str                      # how camera drift was handled
    above_q: np.ndarray
    grasp_q: np.ndarray

    def describe(self) -> dict:
        deg = lambda q: [round(math.degrees(x), 1) for x in q]  # noqa: E731
        return {"name": self.name, "conf": round(self.conf, 2), "grasp_px": [round(x, 1) for x in self.grasp_px],
                "grip_deg": round(self.grip_deg, 1), "width_mm": None if self.width_mm is None else round(self.width_mm, 1),
                "anchor": self.anchor, "above_deg": deg(self.above_q), "grasp_deg": deg(self.grasp_q)}


def wrap180(a: float) -> float:
    return (a + 90.0) % 180.0 - 90.0


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Separating-axis test for two convex polygons (n x 2)."""
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def jaw_zone(center: tuple[float, float], grip_deg: float, half_along: float, half_perp: float) -> np.ndarray:
    """Footprint the open jaws sweep: long across the object (the closing direction), thin along it."""
    g = np.array([math.cos(math.radians(grip_deg)), math.sin(math.radians(grip_deg))])
    p = np.array([-g[1], g[0]])
    c = np.array(center)
    return np.array([c + sa * half_along * g + sp * half_perp * p for sa, sp in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def plan_grasp(det: dict, reach: ReachMap, cfg: PickCfg, joints: list[JointCfg], along: float = 0.0) -> GraspPlan:
    if det.get("conf", 1.0) < cfg.min_conf:
        raise PickError(f"not sure enough what that is (confidence {det['conf']:.2f})")

    u, v = det["center_px"]
    hx, hy = det["heavy_dir"]
    u, v = u + along * det["length_px"] * hx, v + along * det["length_px"] * hy   # toward the handle
    now = (u, v)

    # Camera drift: the arm never stops at tray_view to the pixel. The tray markers show how far it is off.
    shift = reach.anchor_shift(det.get("markers", {}))
    if shift is None:
        anchor = "none (fewer than 2 tray markers visible)"
    else:
        if math.hypot(*shift) > cfg.anchor_max_px:
            raise PickError(f"the camera view is {math.hypot(*shift):.0f}px off its calibration; "
                            "is the arm at tray_view? Re-run `reach look` if the camera or tray moved")
        u, v = u - shift[0], v - shift[1]
        anchor = f"shifted {shift[0]:+.1f},{shift[1]:+.1f}px"

    u0, v0, u1, v1 = reach.bounds(cfg.bounds_margin_px)
    if not (u0 <= u <= u1 and v0 <= v <= v1):
        raise PickError("that is outside the area the arm has been calibrated for")

    width_mm = det["width_px"] * reach.mm_per_px if reach.mm_per_px else None
    if cfg.max_open_mm and width_mm is not None:
        if width_mm > cfg.max_open_mm:
            raise PickError(f"it is {width_mm:.0f} mm across; the jaws open {cfg.max_open_mm:.0f} mm")
        s = reach.mm_per_px
        zone = jaw_zone(now, det["grip_deg"], cfg.max_open_mm / 2 / s, cfg.jaw_width_mm / 2 / s)
        for other in det.get("others", []):
            if _overlap(zone, np.array(other["box_px"], float)):
                raise PickError(f"{other.get('name') or 'another object'} is in the way of the jaws")

    grasp_q, above_q = reach.joints("grasp", u, v), reach.joints("above", u, v)

    # Wrist: the taught angle has the jaws closing along image x; add the object's grip angle.
    # The jaws are symmetric, so +/-180 deg is the same grip: take whichever stays inside joint6's limits.
    j6 = joints[5]
    base = grasp_q[5]
    want = base + cfg.wrist_sign * math.radians(wrap180(det["grip_deg"]))
    ok = [c for c in (want, want - math.pi, want + math.pi) if j6.lower <= c <= j6.upper]
    if not ok:
        raise PickError(f"turning the wrist to {det['grip_deg']:.0f} deg would exceed joint6's limits")
    grasp_q[5] = above_q[5] = min(ok, key=lambda c: abs(c - base))
    return GraspPlan(det.get("name"), det.get("conf", 1.0), (u, v), det["grip_deg"], width_mm, anchor, above_q, grasp_q)
