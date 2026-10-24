"""Reach map: tray-camera pixel -> joint angles, taught instead of computed (no URDF, no IK).

The camera rides on the arm but is only used from one taught pose (`tray_view`), so from there it is a fixed
overhead camera. Fiducial markers are printed on the tray. For each one the operator

    reach look            go to tray_view, record where the camera sees every marker (a fresh session)
    reach touch <id>      jog the jaws onto marker <id> at grasp height, jaws closing along the image x axis
    reach above <id>      raise to hover height over the same marker

and the map is a least-squares fit of joints1-6 over those taught points (affine, quadratic from 9 points).
Joint6 is fitted too, so the taught "jaws along image x" wrist angle is the reference the wrist rotation adds to.

shortcut: assumes the camera looks roughly straight down (angles and scale are not perspective-corrected) and
that the arm repeats `tray_view` to a few pixels; the tray markers re-anchor small drifts at pick time.
Upgrade: URDF + IK, or a full homography with lens distortion.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .config import _atomic_write

QUAD_MIN = 9   # points needed before the quadratic terms are trusted
KINDS = ("grasp", "above")


class ReachError(RuntimeError):
    pass


def _features(z: np.ndarray, quad: bool) -> np.ndarray:
    u, v = z[:, 0], z[:, 1]
    cols = [np.ones_like(u), u, v] + ([u * u, u * v, v * v] if quad else [])
    return np.stack(cols, axis=1)


@dataclass
class _Fit:
    mu: np.ndarray
    sd: np.ndarray
    quad: bool
    w: np.ndarray

    @classmethod
    def make(cls, px: np.ndarray, q: np.ndarray) -> "_Fit":
        if len(px) < 3:
            raise ReachError(f"need at least 3 taught markers, have {len(px)}")
        mu, sd = px.mean(axis=0), px.std(axis=0) + 1e-9
        quad = len(px) >= QUAD_MIN
        f = _features((px - mu) / sd, quad)
        if np.linalg.matrix_rank(f) < f.shape[1]:
            raise ReachError("the taught markers are in a line or too clustered; spread them over the tray")
        return cls(mu, sd, quad, np.linalg.lstsq(f, q, rcond=None)[0])

    def __call__(self, u: float, v: float) -> np.ndarray:
        z = (np.array([[u, v]]) - self.mu) / self.sd
        return (_features(z, self.quad) @ self.w)[0]


@dataclass
class ReachMap:
    path: Path
    view_pose: str = "tray_view"
    mm_per_px: float | None = None
    image: tuple[int, int] | None = None
    lift: np.ndarray | None = None   # hover = grasp + lift (rad), taught once with `reach lift`
    source: str = "aruco"   # "clicked" targets are not seen again at pick time, so they cannot re-anchor drift
    points: dict[int, dict] = field(default_factory=dict)   # id -> {"px": [u, v], "grasp": q, "above": q} (rad)

    def ids(self, kind: str) -> list[int]:
        return sorted(i for i, p in self.points.items() if kind in p)

    def _data(self, kind: str, skip: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        ids = [i for i in self.ids(kind) if i != skip]
        return np.array([self.points[i]["px"] for i in ids], float), np.array([self.points[i][kind] for i in ids], float)

    def ready(self) -> list[str]:
        """What is still missing for picking, in plain words."""
        out = []
        for kind in KINDS:
            n = len(self.ids(kind))
            if n < 3 and not (kind == "above" and self.lift is not None):
                out.append(f"reach map: {n}/3 '{kind}' points taught" + (" (or `reach lift` once)" if kind == "above" else ""))
        return out

    def joints(self, kind: str, u: float, v: float) -> np.ndarray:
        if kind == "above" and len(self.ids("above")) < 3 and self.lift is not None:
            return self.joints("grasp", u, v) + self.lift     # shortcut: same lift everywhere; fine for clearing the table
        px, q = self._data(kind)
        return _Fit.make(px, q)(u, v)

    def bounds(self, margin: float) -> tuple[float, float, float, float]:
        px, _ = self._data("grasp")
        if len(px) == 0:
            raise ReachError("no taught grasp points")
        return (px[:, 0].min() - margin, px[:, 1].min() - margin, px[:, 0].max() + margin, px[:, 1].max() + margin)

    def loo_error_deg(self, kind: str) -> dict[int, float]:
        """Leave-one-out: how far (worst joint, degrees) a marker's joints are from the fit of all the others."""
        out = {}
        for i in self.ids(kind):
            px, q = self._data(kind, skip=i)
            if len(px) < 3:
                continue
            pred = _Fit.make(px, q)(*self.points[i]["px"])
            out[i] = float(np.degrees(np.max(np.abs(pred - self.points[i][kind]))))
        return out

    def anchor_shift(self, seen: dict) -> tuple[float, float] | None:
        """Median image shift of the tray markers now vs when taught (camera drift). None if fewer than 2 visible."""
        if self.source != "aruco":
            return None
        d = [(seen[str(i)][0] - p["px"][0], seen[str(i)][1] - p["px"][1])
             for i, p in self.points.items() if str(i) in seen]
        if len(d) < 2:
            return None
        return statistics.median(x for x, _ in d), statistics.median(y for _, y in d)

    # ── persistence ───────────────────────────────────────────────────────

    def save(self) -> None:
        pts = {}
        for i, p in sorted(self.points.items()):
            pts[i] = {"px": [round(x, 1) for x in p["px"]]}
            for k in KINDS:
                if k in p:
                    pts[i][f"{k}_deg"] = [round(math.degrees(x), 2) for x in p[k]]
        data = {"view_pose": self.view_pose, "mm_per_px": self.mm_per_px,
                "image": list(self.image) if self.image else None, "source": self.source,
                "lift_deg": None if self.lift is None else [round(math.degrees(x), 2) for x in self.lift], "points": pts}
        header = ("# Written by `scrub_arm teach` (reach ...). Marker pixel positions seen from the view pose, and\n"
                  "# the joint angles taught at each marker. The fit is recomputed from these on load.\n")
        _atomic_write(self.path, header + yaml.safe_dump(data, sort_keys=False, default_flow_style=None))


def load_reach(path: Path | str) -> ReachMap:
    path = Path(path)
    rm = ReachMap(path=path)
    if not path.exists():
        return rm
    d = yaml.safe_load(path.read_text()) or {}
    rm.view_pose = d.get("view_pose", rm.view_pose)
    rm.mm_per_px = d.get("mm_per_px")
    rm.image = tuple(d["image"]) if d.get("image") else None
    rm.source = d.get("source", "aruco")
    rm.lift = np.radians(np.array(d["lift_deg"], float)) if d.get("lift_deg") else None
    for i, p in (d.get("points") or {}).items():
        pt = {"px": [float(x) for x in p["px"]]}
        for k in KINDS:
            if f"{k}_deg" in p:
                pt[k] = np.radians(np.array(p[f"{k}_deg"], float))
        rm.points[int(i)] = pt
    return rm


# ── teach-CLI commands ────────────────────────────────────────────────────

REACH_HELP = """\
  reach look              go to the view pose and record where the camera sees each tray marker (starts over)
  reach scale <mm>        with a tool of known length (mm) lying in view: sets mm per pixel (not needed with printed markers)
  reach touch <id>        record the current setpoint as the GRASP pose over marker <id>
                          (jaws open around a typical object's mid-thickness, closing direction along the image x axis)
  reach above <id>        record the current setpoint as the HOVER pose over marker <id>
  reach lift <id>         shortcut for `above`: after `reach touch <id>`, raise the arm to a hover height and type this
                          once; every target then hovers by the same joint offset (skip the other `above` steps)
  reach check             how many points are taught and how well they agree (leave-one-out, degrees)"""


def reach_cmd(app, args: list[str]) -> str:
    rm, d, pc = app.reach, app.driver, app.cfg.pick
    sub = args[0] if args else "check"
    if sub == "look":
        if app.vision is None:
            return "pick.vision_url is not set in arm.yaml"
        d.clear_abort()
        d.move_pose(pc.view_pose)
        info = app.vision.markers()
        found = info["markers"]
        if len(found) < 3:
            return (f"camera sees {len(found)} markers ({sorted(found)}); need at least 3. Print some "
                    "(python -m scrub_vision.markers --sheet) or click targets (python -m scrub_vision.targets --camera 0)")
        sides = [m["side_px"] for m in found.values() if m.get("side_px")]
        rm.view_pose, rm.image, rm.source = pc.view_pose, tuple(info["image"]), info.get("source", "aruco")
        rm.mm_per_px = pc.marker_mm / statistics.median(sides) if pc.marker_mm and sides else None
        rm.points = {int(i): {"px": m["center"]} for i, m in found.items()}
        rm.save()
        scale = f", {rm.mm_per_px:.3f} mm/px" if rm.mm_per_px else " (no scale yet: `reach scale <mm>`)"
        return f"saw {rm.source} targets {sorted(rm.points)}{scale}. Now `reach touch <id>` and `reach above <id>` for each."
    if sub == "scale":
        if app.vision is None:
            return "pick.vision_url is not set in arm.yaml"
        det = app.vision.detect(None, 3)
        if not det.get("found"):
            return "the camera sees no object; put a tool of known length on the mat, then retry"
        rm.mm_per_px = float(args[1]) / det["length_px"]
        rm.save()
        return f"{det['length_px']:.0f} px = {float(args[1]):g} mm -> {rm.mm_per_px:.3f} mm/px"
    if sub == "lift":
        mid = int(args[1])
        if mid not in rm.points or "grasp" not in rm.points[mid]:
            return f"`reach touch {mid}` first, then raise the arm and type `reach lift {mid}`"
        if not d.enabled:
            return "motors are limp; `hold` first so the setpoint is what gets saved"
        q = d.q_command()
        d.check_limits(q)
        rm.lift = q - rm.points[mid]["grasp"]
        rm.lift[5] = 0.0     # the wrist angle comes from the object, not the hover offset
        rm.save()
        return "hover offset (deg): " + " ".join(f"{math.degrees(x):+.1f}" for x in rm.lift[:5])
    if sub in ("touch", "above"):
        mid = int(args[1])
        if mid not in rm.points:
            return f"marker {mid} was not seen by `reach look` (saw {sorted(rm.points)})"
        if not d.enabled:
            return "motors are limp; `hold` first so the setpoint is what gets saved"
        q = d.q_command()
        d.check_limits(q)
        rm.points[mid]["grasp" if sub == "touch" else "above"] = q
        rm.save()
        return f"marker {mid} {sub}: {len(rm.ids('grasp'))} grasp / {len(rm.ids('above'))} above points taught"
    if sub == "check":
        lines = [f"markers seen: {sorted(rm.points)}", *(rm.ready() or ["enough points taught"])]
        for kind in KINDS:
            err = rm.loo_error_deg(kind)
            if err:
                lines.append(f"{kind}: leave-one-out worst joint error "
                             + ", ".join(f"#{i}={e:.1f}" for i, e in err.items())
                             + f"  ({'quadratic' if len(rm.ids(kind)) >= QUAD_MIN else 'affine'} fit)")
        return "\n".join(lines)
    return REACH_HELP
