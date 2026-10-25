"""Our 8 printed (red) parts on the white paper: find each one, its outline and grab point, and which part it is.

Fitted and tested on real wrist-camera frames (hw_check/real/): see config/red_parts.yaml for the numbers.

1. Colour: Lab a* hysteresis (strong red seeds a* > 28 grown into connected a* > 16: the pale, lit ends). The wood
   reads a* 8-17 and never touches a seed, the paper reads ~ -9.
2. Each pixel goes to a metric top-down table canvas through the arm's pose (FK + hand-eye + K), so the shape
   measurements don't depend on where the camera is.
3. A part cut off by the frame edge or the gripper fingers is reported as incomplete (no identity: a cut scalpel
   looks like a shorter one).
4. Identity: needle holders have ring holes; then the nearest class in (length, mean width, median width, width of
   the narrower end, that end / median width), each in units of that class's spread. The two micro scissors look
   the same (user: they may share a type); flat and on its side they look different, so each rest is its own class.
5. Grab point: the centre of the part's oriented bounding box (user: pick at the bounding box's centre), jaws
   across the long axis.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from .identify import Detection

ROOT = Path(__file__).resolve().parents[2]
MODEL_YAML = Path(__file__).resolve().parents[1] / "config" / "red_parts.yaml"
FINGERS_PNG = Path(__file__).resolve().parents[1] / "config" / "finger_mask.png"
A_STRONG, A_WEAK = 28, 16
MM_PX = 0.5                               # canvas resolution
WINDOW_MM = (150.0, 750.0, -350.0, 350.0)  # table x0, x1, y0, y1 the canvas covers
PART_Z_MM = 4.0                           # project at this height above the table (parts are 3-12 mm thick)
UNSEEN_MARGIN_PX = 12                     # a part this close to the frame edge or a finger counts as cut off
MIN_AREA_MM2 = 150
NB = 24                                   # width profile bins along the part
REJECT_SIGMA = 4.0                        # farther than this from every class: not one of our parts
MARGIN_SIGMA = 0.5                        # runner-up closer than this: report both as candidates
SAME_PART = {"micro_scissors": ("scissors_micro_forward", "scissors_micro_back")}


# ── geometry ──────────────────────────────────────────────────────────────
class Camera:
    """Wrist camera on the arm: pixel <-> table mm for a joint vector (camera.yaml K, dist, hand-eye, table z)."""

    def __init__(self, cam: dict | None = None) -> None:
        cam = cam or yaml.safe_load((ROOT / "config" / "camera.yaml").read_text())
        self.K, self.dist = np.array(cam["K"], float), np.array(cam["dist"], float)
        self.p_tc = np.array(cam["handeye"]["p_tool_cam_m"], float)
        self.R_tc = np.array(cam["handeye"]["R_tool_cam"], float)
        self.table_z = cam["table_z_mm"] / 1000
        self._kin = None

    def pose(self, q) -> tuple[np.ndarray, np.ndarray]:
        if self._kin is None:
            import sys
            sys.path.insert(0, str(ROOT / "arm"))
            from scrub_arm.kinematics import Kinematics
            self._kin = Kinematics()
        p, R = self._kin.fk(np.asarray(q, float))
        return p + R @ self.p_tc, R @ self.R_tc

    def px_to_mm(self, px, q, z_mm: float = PART_Z_MM) -> np.ndarray:
        """(N, 2) raw (distorted) pixels -> (N, 2) table mm at z_mm above the table."""
        c, R = self.pose(q)
        n = cv2.undistortPoints(np.asarray(px, float).reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        d = np.column_stack([n, np.ones(len(n))]) @ R.T
        t = (self.table_z + z_mm / 1000 - c[2]) / d[:, 2]
        return (c + t[:, None] * d)[:, :2] * 1000

    def mm_to_px(self, mm, q, z_mm: float = PART_Z_MM) -> np.ndarray:
        c, R = self.pose(q)
        P = np.column_stack([np.asarray(mm, float) / 1000, np.full(len(mm), self.table_z + z_mm / 1000)])
        rvec = cv2.Rodrigues(R.T)[0]
        return cv2.projectPoints(P, rvec, -R.T @ c, self.K, self.dist)[0].reshape(-1, 2)

    def H_px_to_mm(self, q, shape) -> np.ndarray:
        """A homography fitted to px_to_mm over the frame (for code that wants one, e.g. planner.feasible)."""
        h, w = shape[:2]
        g = np.stack(np.meshgrid(np.linspace(0, w - 1, 12), np.linspace(0, h - 1, 8)), -1).reshape(-1, 2)
        return cv2.findHomography(g, self.px_to_mm(g, q))[0]


class _Canvas:
    """Remap tables raw image -> top-down canvas for one pose (cached: the scan pose repeats)."""

    def __init__(self, cam: Camera, q, shape) -> None:
        x0, x1, y0, y1 = WINDOW_MM
        self.w, self.h = int((y1 - y0) / MM_PX), int((x1 - x0) / MM_PX)
        rows, cols = np.mgrid[0:self.h, 0:self.w]
        mm = np.column_stack([(x1 - rows * MM_PX).ravel(), (y1 - cols * MM_PX).ravel()])  # far side up, +y left
        c, R = cam.pose(q)
        px = cam.mm_to_px(mm, q).reshape(self.h, self.w, 2).astype(np.float32)
        P = np.c_[mm / 1000, np.full(len(mm), cam.table_z + PART_Z_MM / 1000)]
        depth = (P - c) @ R[:, 2]
        px[(depth <= 0.05).reshape(self.h, self.w)] = -1             # behind (or right under) the camera
        self.map = px
        self.shape = shape[:2]

    def warp(self, img, interp=cv2.INTER_NEAREST):
        return cv2.remap(img, self.map[..., 0], self.map[..., 1], interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def to_mm(self, rc: np.ndarray) -> np.ndarray:
        x0, x1, y0, y1 = WINDOW_MM
        return np.column_stack([x1 - rc[:, 0] * MM_PX, y1 - rc[:, 1] * MM_PX])

    def back_to_image(self, comp: np.ndarray) -> np.ndarray:
        """Canvas component -> raw image mask (splat each canvas pixel's image position, then close holes)."""
        m = np.zeros(self.shape, np.uint8)
        r, c = np.nonzero(comp)
        p = np.round(self.map[r, c]).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 1] >= 0) & (p[:, 0] < self.shape[1]) & (p[:, 1] < self.shape[0])
        m[p[ok, 1], p[ok, 0]] = 1
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)


# ── segmentation ──────────────────────────────────────────────────────────
def red_mask(bgr: np.ndarray) -> np.ndarray:
    a = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 1].astype(np.int16) - 128
    k = np.ones((3, 3), np.uint8)
    strong = cv2.morphologyEx((a > A_STRONG).astype(np.uint8), cv2.MORPH_OPEN, k)
    n, lab = cv2.connectedComponents((a > A_WEAK).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[strong > 0])] = True
    keep[0] = False
    return cv2.morphologyEx(keep[lab].astype(np.uint8), cv2.MORPH_OPEN, k)


def load_fingers(shape) -> np.ndarray:
    """The open fingers in the image (camera rides on the gripper, so they're always at the same pixels)."""
    f = cv2.imread(str(FINGERS_PNG), 0)
    if f is None:
        return np.zeros(shape[:2], bool)
    return cv2.resize(f, shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0


# ── features + identity ───────────────────────────────────────────────────
def features(comp: np.ndarray) -> dict:
    r, c = np.nonzero(comp)
    P = np.column_stack([c, r]).astype(float) * MM_PX
    mu = P.mean(0)
    w, v = np.linalg.eigh(np.cov((P - mu).T))
    ax = v[:, np.argmax(w)]
    u, t = (P - mu) @ ax, (P - mu) @ np.array([-ax[1], ax[0]])
    L = float(u.max() - u.min())
    bins = np.clip(((u - u.min()) / L * NB).astype(int), 0, NB - 1)
    fill = np.bincount(bins, minlength=NB) * MM_PX ** 2 / (L / NB)    # material width per bin, mm
    cs, hier = cv2.findContours(comp.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    holes = sum(1 for i, h in enumerate(hier[0]) if h[3] >= 0 and cv2.contourArea(cs[i]) * MM_PX ** 2 > 30)
    area = float(comp.sum() * MM_PX ** 2)
    medw = float(np.median(fill))
    tip = float(min(fill[1:6].mean(), fill[-6:-1].mean()))            # the narrower end
    # oriented bounding box centre (canvas px) and long axis
    mid_u, mid_t = (u.max() + u.min()) / 2, (t.max() + t.min()) / 2
    centre = (mu + mid_u * ax + mid_t * np.array([-ax[1], ax[0]])) / MM_PX
    jaw = np.abs(u - mid_u) < 6.0                                       # under the ~12 mm wide jaws
    grip_w = float(t[jaw].max() - t[jaw].min()) if jaw.any() else medw  # what the jaws must span there
    return {"length": L, "area": area, "holes": holes, "medw": medw, "tipw": tip, "fill": fill.tolist(),
            "vec": [L, area / L, medw, tip, tip / medw], "grip_w": grip_w, "centre_rc": (centre[1], centre[0]), "axis_cr": ax}


def classify(model: dict, f: dict) -> list[tuple[float, str]]:
    """[(distance in sigmas, class)] nearest first, among classes with the same ring-ness."""
    x, ringed = np.array(f["vec"]), f["holes"] >= 2
    return sorted((float(np.sqrt(np.mean(((x - np.array(m["mu"])) / np.array(m["sd"])) ** 2))), lab)
                  for lab, m in model["classes"].items() if m["ringed"] == ringed)


def part_of(cls: str) -> str:
    return cls.split("@")[0]               # sub-modes, e.g. micro_scissors@flat


# ── detector ──────────────────────────────────────────────────────────────
class RedPartsDetector:
    """detect(bgr, q) -> [Detection] in table mm (base frame), one per red blob, identity None if cut off/unknown."""

    def __init__(self, model_yaml: Path = MODEL_YAML, cam: Camera | None = None) -> None:
        self.model = yaml.safe_load(Path(model_yaml).read_text())
        self.cam = cam or Camera()
        self._canvas: tuple[tuple, _Canvas] | None = None
        from .templates import load_instruments
        self.instruments = load_instruments()

    def canvas(self, q, shape) -> _Canvas:
        key = (tuple(np.round(np.asarray(q, float), 4)), shape[:2])
        if self._canvas is None or self._canvas[0] != key:
            self._canvas = (key, _Canvas(self.cam, q, shape))
        return self._canvas[1]

    def detect(self, bgr: np.ndarray, q) -> list[Detection]:
        cv_ = self.canvas(q, bgr.shape)
        seen = np.ones(bgr.shape[:2], np.uint8)
        seen[load_fingers(bgr.shape)] = 0
        seen = cv2.erode(seen, np.ones((2 * UNSEEN_MARGIN_PX + 1,) * 2, np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0)
        m, valid = cv_.warp(red_mask(bgr)), cv_.warp(seen)
        m = cv2.morphologyEx(m & valid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        border = cv2.dilate((valid == 0).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        dets = []
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] * MM_PX ** 2 < MIN_AREA_MM2:
                continue
            comp = lab == i
            dets.append(self._detection(comp, not (comp & border).any(), cv_))
        return dets

    def _detection(self, comp: np.ndarray, complete: bool, cv_: _Canvas) -> Detection:
        f = features(comp)
        g_mm = cv_.to_mm(np.array([f["centre_rc"]]))[0] + np.array(self.model.get("grasp_offset_mm", (0.0, 0.0)))
        ax_cr = f["axis_cr"]                                   # canvas (col, row) -> table: col = -y, row = -x
        axis_deg = float(np.degrees(np.arctan2(-ax_cr[0], -ax_cr[1])))
        det = Detection(mask=cv_.back_to_image(comp), coarse=None, fine_id=None, length_mm=f["length"],
                        width_mm=f["medw"], axis_deg=axis_deg, grasp_mm=(float(g_mm[0]), float(g_mm[1])),
                        yaw_deg=axis_deg + 90.0, exposure=1.0 if complete else 0.0)
        det.shape_features = f                                # for fitting (scripts/fit_red_parts.py)
        vs, us = np.nonzero(det.mask)
        if len(us):
            det.centroid_px = (float(us.mean()), float(vs.mean()))
        if not complete:
            return self._partial(det, f)
        ranked = classify(self.model, f)
        if not ranked or ranked[0][0] > REJECT_SIGMA:
            return det                                        # red, but not one of our parts
        best_d, best = ranked[0]
        part = part_of(best)
        ids = list(SAME_PART.get(part, (part,)))
        det.fine_id, det.alt_ids = ids[0], ids[1:]
        det.coarse = self.instruments[ids[0]].coarse if ids[0] in self.instruments else part
        rivals = [part_of(c) for d, c in ranked[1:] if d - best_d < MARGIN_SIGMA and part_of(c) != part]
        det.candidates = sorted(set(ids) | {i for r in rivals for i in SAME_PART.get(r, (r,))})
        if rivals:                                            # too close to call: don't claim an identity
            det.fine_id, det.alt_ids = None, []
        det.confidence = float(max(0.0, 1.0 - best_d / (2 * REJECT_SIGMA)))
        det.thickness_mm = self.instruments[ids[0]].thickness_mm if ids[0] in self.instruments else 5.0
        return det

    def _partial(self, det: Detection, f: dict) -> Detection:
        """Cut off by a finger or the frame: never a sure identity. Candidates: parts long enough to be cut down to
        this, matching on width alone; exposure = visible length / that part's length (e.g. the #20 scalpel at the
        scan pose just touches the right finger: 173 of 171 mm visible -> exposure 1, 'I think this is ...')."""
        x, ringed = np.array(f["vec"]), f["holes"] >= 2
        ranked = sorted((float(np.sqrt(np.mean(((x[1:] - np.array(m["mu"][1:])) / np.array(m["sd"][1:])) ** 2))), c)
                        for c, m in self.model["classes"].items()
                        if m["ringed"] == ringed and m["mu"][0] >= x[0] - 2 * m["sd"][0])
        if not ranked or ranked[0][0] > REJECT_SIGMA:
            return det
        best_d, best = ranked[0]
        near = [part_of(c) for d, c in ranked if d - best_d < MARGIN_SIGMA]
        det.candidates = sorted({i for p in near for i in SAME_PART.get(p, (p,))})
        mu_len = self.model["classes"][best]["mu"][0]
        det.exposure = float(min(1.0, x[0] / mu_len))
        det.confidence = float(max(0.0, 1.0 - best_d / (2 * REJECT_SIGMA)))
        fid = SAME_PART.get(part_of(best), (part_of(best),))[0]
        det.coarse = self.instruments[fid].coarse if fid in self.instruments else part_of(best)
        det.thickness_mm = self.instruments[fid].thickness_mm if fid in self.instruments else 5.0
        return det
