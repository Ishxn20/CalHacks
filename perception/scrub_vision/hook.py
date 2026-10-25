"""Tell the two micro scissors apart by the hook at the tip (user: check it at the 10 cm hover).

From above, lying on their side, the forward-facing (45 deg) scissors end in a short straight bend and the back-facing
(135 deg) ones in a hook that curls back; elsewhere their outlines are the same (95 % overlap). So: render both
side-on outlines from the STLs at the image's scale, line each up with the mask at the tip end (both ends, both
mirror images, small slides and turns), and score only the last TIP_MM of the outline. The clearly better one wins;
if the tip isn't in view (cut by the frame edge or the gripper) or the scores are close, the answer is None.
"""
from __future__ import annotations

import cv2
import numpy as np

from .templates import load_instruments, mesh_for, silhouette

IDS = ("scissors_micro_forward", "scissors_micro_back")
TIP_MM = 30.0           # where the two differ: the bend/hook and the joint knob behind it
MARGIN = 0.08           # tip-zone IoU the better one must win by
BORDER_PX = 3           # a mask touching the frame edge / gripper this close is cut off there


def _frame(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = pts.mean(0)
    w, v = np.linalg.eigh(np.cov((pts - c).T))
    return c, v[:, np.argmax(w)]


def _tip_zone_iou(mask: np.ndarray, tmpl: np.ndarray, px_per_mm: float, tip_at: np.ndarray, axis: np.ndarray,
                  shift: tuple[float, float], turn: float) -> float:
    """Template (tip at +x, centreline at mid-height) placed with its tip end on `tip_at`, its +x along `axis`;
    IoU with the mask inside the template's last TIP_MM (both dilated by the zone)."""
    h, w = tmpl.shape
    ys = np.nonzero(tmpl[:, -1 - int(round(2 * px_per_mm)):].any(1))[0]   # the very tip's rows
    t_tip = np.array([w - 1.0, ys.mean() if len(ys) else h / 2])
    a = np.arctan2(axis[1], axis[0]) + turn
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    off = tip_at + R @ np.array(shift) - R @ t_tip
    M = np.c_[R, off].astype(np.float32)
    H, W = mask.shape
    placed = cv2.warpAffine(tmpl, M, (W, H), flags=cv2.INTER_NEAREST) > 0
    zone_t = np.zeros_like(tmpl)
    zone_t[:, max(0, int(w - TIP_MM * px_per_mm)):] = 255
    zone = cv2.warpAffine(zone_t, M, (W, H), flags=cv2.INTER_NEAREST) > 0
    inter = (placed & mask & zone).sum()
    union = ((placed | mask) & zone).sum()
    return inter / union if union else 0.0


_tmpl_cache: dict = {}


def _templates(px_per_mm: float) -> dict[str, list[np.ndarray]]:
    key = round(px_per_mm, 2)
    if key not in _tmpl_cache:
        insts = load_instruments()
        _tmpl_cache[key] = {fid: [silhouette(mesh_for(insts[fid], "side"), 1.0 / px_per_mm, flip)
                                  for flip in (False, True)] for fid in IDS}
    return _tmpl_cache[key]


def hook_type(mask: np.ndarray, px_per_mm: float | None = None,
              blocked: np.ndarray | None = None) -> tuple[str | None, dict[str, float]]:
    """(scissors_micro_forward | scissors_micro_back | None, tip-zone score per id) for one micro scissors' mask.

    px_per_mm: the image scale at the instrument (from the camera height); None = from the mask's length, which needs
    the whole instrument in view. blocked: pixels nothing can be seen through (the gripper)."""
    mask = mask.astype(bool)
    vs, us = np.nonzero(mask)
    if len(us) < 200:
        return None, {}
    pts = np.c_[us, vs].astype(float)
    c, axis = _frame(pts)
    t = (pts - c) @ axis
    length_px = float(np.ptp(t))
    k = np.ones((2 * BORDER_PX + 1,) * 2, np.uint8)
    edge = np.zeros_like(mask)
    edge[:BORDER_PX], edge[-BORDER_PX:], edge[:, :BORDER_PX], edge[:, -BORDER_PX:] = True, True, True, True
    if blocked is not None:
        edge |= cv2.dilate(blocked.astype(np.uint8), k).astype(bool)
    cut = mask & edge
    scales = ([px_per_mm * f for f in (0.92, 1.0, 1.08)] if px_per_mm else
              [length_px / 150.0 * f for f in (0.95, 1.0, 1.05)])
    best = {fid: 0.0 for fid in IDS}
    for e in (1.0, -1.0):                                   # either end could be the tip
        i_tip = int(np.argmax(t * e))
        tip_at = pts[i_tip]
        if cut.any():                                       # the tip end must be fully visible
            cvs, cus = np.nonzero(cut)
            if np.min(np.hypot(cus - tip_at[0], cvs - tip_at[1])) < TIP_MM * scales[1]:
                continue
        if not px_per_mm and cut.any():
            continue                                        # length (so scale) unknown when cut off
        for s in scales:
            for fid, tms in _templates(s).items():
                for tm in tms:
                    for turn in np.radians([-4.0, 0.0, 4.0]):
                        for dx in (-3, 0, 3):
                            for dy in (-3, 0, 3):
                                v = _tip_zone_iou(mask, tm, s, tip_at, axis * e, (dx, dy), turn)
                                best[fid] = max(best[fid], v)
    f, b = best[IDS[0]], best[IDS[1]]
    if max(f, b) < 0.5 or abs(f - b) < MARGIN:
        return None, best
    return (IDS[0] if f > b else IDS[1]), best
