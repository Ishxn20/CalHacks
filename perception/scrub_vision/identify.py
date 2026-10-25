"""Identify instruments from instance masks: PCA axis, metric length gate, template IoU, grasp pose.

All masks are in undistorted scan-pose pixels; `H` maps those pixels to table (x, y) mm.
Grasp yaw convention (matches scrub_arm Kinematics.ik_topdown): the direction the jaws OPEN,
i.e. across the instrument's shank = axis + 90 deg.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .templates import Instrument, Template, interchangeable

LENGTH_TOL = 0.15   # real camera: a lone printed scalpel measured 9-11 % short (thin pale blade tip, mapping near the frame edge); was 0.08
MARGIN_MIN = 0.05
FIT_MIN = 0.6           # below this the best template doesn't explain the blob: identity unknown
CANON_MM_PER_PX = 0.5
TOL_MM = 1.5            # silhouette match tolerance (thin tines are ~2 mm wide)
LENGTH_WEIGHT = 2.0     # score penalty per unit relative length error (Castroviejo 140 vs micro scissors 150)


def bbox_centre(mask: np.ndarray) -> tuple[float, float]:
    """Centre of the mask's bounding box (pixels): the pickup point (user: pick at the bounding box's centre)."""
    vs, us = np.nonzero(mask)
    return (float(us.min() + us.max()) / 2, float(vs.min() + vs.max()) / 2)


@dataclass(eq=False)   # identity equality: masks are arrays
class Detection:
    mask: np.ndarray
    coarse: str | None
    fine_id: str | None
    alt_ids: list[str] = field(default_factory=list)   # interchangeable instruments (scalpels, micro scissors)
    candidates: list[str] = field(default_factory=list)  # fine ids this could be (all within MARGIN of best)
    length_mm: float = 0.0
    width_mm: float = 0.0
    axis_deg: float = 0.0
    centroid_px: tuple[float, float] = (0.0, 0.0)
    grasp_px: tuple[float, float] = (0.0, 0.0)
    grasp_mm: tuple[float, float] = (0.0, 0.0)
    yaw_deg: float = 0.0
    thickness_mm: float = 5.0
    exposure: float = 0.0
    confidence: float = 0.0
    seg_conf: float = 1.0


def _apply(H: np.ndarray, pts) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, np.float64))
    h = np.c_[pts, np.ones(len(pts))] @ H.T
    return h[:, :2] / h[:, 2:]


def _contours(mask: np.ndarray):
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return [], []
    outer = [c.reshape(-1, 2).astype(np.float64) for c, h in zip(cs, hier[0]) if h[3] == -1]
    holes = [c.reshape(-1, 2).astype(np.float64) for c, h in zip(cs, hier[0]) if h[3] != -1]
    return outer, holes


def _frame(pts_mm: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts_mm.mean(0)
    cov = np.cov((pts_mm - c).T)
    w, v = np.linalg.eigh(cov)
    major = v[:, np.argmax(w)]
    return c, float(np.arctan2(major[1], major[0]))


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _canon_mask(outer, holes, c, theta, size_mm: float) -> np.ndarray:
    """Rasterise polygons (mm) into a canvas centred on c, rotated so theta -> +x."""
    n = int(size_mm / CANON_MM_PER_PX)
    img = np.zeros((n, n), np.uint8)
    R = _rot(-theta)

    def to_px(p):
        return np.round((p - c) @ R.T / CANON_MM_PER_PX + n / 2).astype(np.int32)

    cv2.fillPoly(img, [to_px(p) for p in outer], 255)
    if holes:
        cv2.fillPoly(img, [to_px(p) for p in holes], 0)
    return img > 0


def measure(mask: np.ndarray, H: np.ndarray) -> tuple[float, float, float, tuple[float, float]]:
    """(length_mm, width_mm, axis_deg, centroid_px) of a mask via PCA in table mm."""
    vs, us = np.nonzero(mask)
    mm = _apply(H, np.c_[us, vs])
    c, theta = _frame(mm)
    q = (mm - c) @ _rot(-theta).T
    length = float(np.ptp(q[:, 0]))
    width = float(np.ptp(q[:, 1]))
    cpx = _apply(np.linalg.inv(H), c)[0]
    return length, width, float(np.degrees(theta)), (float(cpx[0]), float(cpx[1]))


class _TemplateCanon:
    """Template polygons in mm, with its own PCA frame, cached per template."""

    def __init__(self, t: Template) -> None:
        outer, holes = _contours(t.mask > 0)
        s = t.mm_per_px
        self.outer = [p * s for p in outer]
        self.holes = [p * s for p in holes]
        vs, us = np.nonzero(t.mask)
        self.c, self.theta = _frame(np.c_[us, vs] * s)
        self.grasp_q = (np.array(t.grasp_px) * s - self.c) @ _rot(-self.theta).T
        self.tip_q = (np.array([t.mask.shape[1], t.grasp_px[1]]) * s - self.c) @ _rot(-self.theta).T


def _tc(t: Template) -> _TemplateCanon:
    if t._canon is None:
        t._canon = _TemplateCanon(t)
    return t._canon


def identify(mask: np.ndarray, coarse: str | None, H: np.ndarray, templates: list[Template],
             instruments: dict[str, Instrument], seg_conf: float = 1.0) -> Detection:
    length, width, axis_deg, cpx = measure(mask, H)
    det = Detection(mask=mask, coarse=coarse, fine_id=None, length_mm=length, width_mm=width,
                    axis_deg=axis_deg, centroid_px=cpx, seg_conf=seg_conf)
    # pickup point: the bounding box's centre (user); jaws across the long axis until a template gives the tip end
    bc = bbox_centre(mask)
    det.grasp_px, det.grasp_mm = bc, tuple(float(v) for v in _apply(H, bc)[0])
    det.yaw_deg = axis_deg + 90.0
    outer_px, holes_px = _contours(mask)
    if not outer_px:
        return det
    outer = [_apply(H, p) for p in outer_px]
    holes = [_apply(H, p) for p in holes_px]
    vs, us = np.nonzero(mask)
    c, theta = _frame(_apply(H, np.c_[us, vs]))
    size = max(length, 60.0) * 1.4 + 40
    cands = [t for t in templates if (coarse is None or t.coarse == coarse)
             and abs(length - t.length_mm) <= LENGTH_TOL * t.length_mm]
    k = np.ones((2 * int(TOL_MM / CANON_MM_PER_PX) + 1,) * 2, np.uint8)
    scored = []   # (score, fit, exposure, template, end_flip)
    det_masks = {e: _canon_mask(outer, holes, c, theta + e * np.pi, size) for e in (0, 1)}
    det_dil = {e: cv2.dilate(m.astype(np.uint8), k).astype(bool) for e, m in det_masks.items()}
    for t in cands:
        tc = _tc(t)
        tm = _canon_mask(tc.outer, tc.holes, tc.c, tc.theta, size)
        tm_dil = cv2.dilate(tm.astype(np.uint8), k).astype(bool)
        for e, dm in det_masks.items():
            fit = (dm & tm_dil).sum() / max(dm.sum(), 1)            # blob pixels the template explains
            exposure = (tm & det_dil[e]).sum() / max(tm.sum(), 1)   # template pixels that are visible
            score = min(fit, exposure) * 0.5 + fit * 0.5 - LENGTH_WEIGHT * abs(length - t.length_mm) / t.length_mm
            scored.append((score, fit, exposure, t, e))
    if not scored:
        return det
    scored.sort(key=lambda s: -s[0])
    best, fit, exposure, bt, be = scored[0]
    looks = interchangeable(instruments, bt.fine_id)
    rival = next((s[0] for s in scored if s[3].fine_id not in looks), 0.0)
    det.exposure = float(exposure)
    det.confidence = float(best * seg_conf)
    det.thickness_mm = bt.thickness_mm          # as it lies (a forceps on its side is its width tall)
    det.coarse = bt.coarse if coarse is None else coarse
    if fit >= FIT_MIN:
        det.candidates = sorted({s[3].fine_id for s in scored if best - s[0] < MARGIN_MIN} | looks)
        if best - rival >= MARGIN_MIN:
            det.fine_id = bt.fine_id
            det.alt_ids = sorted(looks - {bt.fine_id})
    tc = _tc(bt)
    th = theta + be * np.pi
    grasp_mm = tc.grasp_q @ _rot(th).T + c          # the template's grasp point: only for which way the tip points
    tip_mm = tc.tip_q @ _rot(th).T + c
    tip_dir = np.degrees(np.arctan2(*(tip_mm - grasp_mm)[::-1]))
    det.axis_deg = float(tip_dir)
    det.yaw_deg = float(tip_dir + 90.0)
    return det


def above(a: np.ndarray, b: np.ndarray, tol_px: int = 4) -> bool:
    """Is mask a lying on top of mask b? Where they cross, the upper mask is continuous and the lower is cut."""
    k = np.ones((2 * tol_px + 1,) * 2, np.uint8)
    da = cv2.dilate(a.astype(np.uint8), k).astype(bool)
    db = cv2.dilate(b.astype(np.uint8), k).astype(bool)
    zone = da & db
    if zone.sum() < 4 * tol_px * tol_px:
        return False                       # they don't touch
    cov_a, cov_b = (a & zone).sum(), (b & zone).sum()
    return cov_a > 1.3 * cov_b


def on_top(det: Detection, others: list[Detection]) -> bool:
    return not any(above(o.mask, det.mask) for o in others if o is not det)


def feasible(det: Detection, others: list[Detection], H: np.ndarray,
             jaw_w_mm: float = 12.0, jaw_open_mm: float = 40.0) -> bool:
    """Can the open jaws come down at det's grasp point without landing on an instrument that's on top?"""
    a = np.radians(det.axis_deg)
    along, across = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    g = np.array(det.grasp_mm)
    corners = [g + sa * along * jaw_w_mm / 2 + sb * across * jaw_open_mm / 2
               for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    poly = np.round(_apply(np.linalg.inv(H), np.array(corners))).astype(np.int32)
    jaw = np.zeros(det.mask.shape, np.uint8)
    cv2.fillPoly(jaw, [poly], 1)
    jaw = jaw.astype(bool) & ~det.mask
    return not any((jaw & o.mask).any() and above(o.mask, det.mask) for o in others if o is not det)


def segment_background(bgr: np.ndarray, gripper_mask: np.ndarray | None = None, min_delta: float = 26.0,
                       min_area_px: int = 300) -> list[np.ndarray]:
    """Fallback segmenter, any instrument colour: pixels that differ from the table surface.

    The table colour is estimated per region (median over a coarse grid, so vignetting and uneven light
    don't count as objects). Weak backup only: on synthetic scenes with random instrument colours and
    textured backgrounds it finds ~2/3 of instruments, misses ones close to the table colour, and merges
    touching instruments. The trained model is the real segmenter.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    small = cv2.resize(lab, (16, 9), interpolation=cv2.INTER_AREA)
    # robust table estimate: per channel median of the coarse grid, blended with the cell's own value
    med = np.median(small.reshape(-1, 3), 0)
    cell = np.where(np.linalg.norm(small - med, axis=2, keepdims=True) < min_delta, small, med)
    table = cv2.resize(cell, (w, h), interpolation=cv2.INTER_CUBIC)
    m = (np.linalg.norm(lab - table, axis=2) > min_delta).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if gripper_mask is not None:
        m[gripper_mask] = 0
    n, lab_ids, stats, _ = cv2.connectedComponentsWithStats(m)
    return [lab_ids == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area_px]


def annotate(bgr: np.ndarray, dets: list[Detection], highlight: Detection | None = None) -> np.ndarray:
    out = bgr.copy()
    rng = np.random.default_rng(1)
    for d in dets:
        col = (0, 0, 255) if d is highlight else tuple(int(x) for x in rng.integers(60, 255, 3))
        cs, _ = cv2.findContours(d.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, col, 2)
        u, v = map(int, d.grasp_px)
        cv2.drawMarker(out, (u, v), col, cv2.MARKER_CROSS, 16, 2)
        label = f"{d.fine_id or d.coarse or '?'} {d.length_mm:.0f}mm e{d.exposure:.2f}"
        cv2.putText(out, label, (int(d.centroid_px[0]), int(d.centroid_px[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, col, 2)
    return out
