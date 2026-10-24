"""Wrist-camera object finding: camera capture, click-to-learn colour detection, calibration storage.

Detection is classic colour segmentation (plan step 01): click the object in the live view,
its colour range is learned, and each frame the largest blob of that colour is the object.
The object's centre is the centroid of the blob's convex hull, so a ring (rubber band)
reports its middle, which is where the jaws should close.
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .config import DEFAULT_CONFIG_DIR, _atomic_write

DEFAULT_VISION_PATH = DEFAULT_CONFIG_DIR / "vision.yaml"


@dataclass
class Detection:
    center: tuple[float, float]   # pixel (u, v)
    area: float                   # px^2 of the filled blob
    hull: np.ndarray              # contour points for drawing
    angle_deg: float              # image direction of the blob's long axis (0 = image right, 90 = image down)
    aspect: float                 # long side / short side of the blob's bounding rectangle (1 = round)


@dataclass
class VisionCfg:
    path: Path
    camera_index: int = 0
    # Learned colour, OpenCV HSV (H 0-179). None until you click the object.
    hsv_lo: list[int] | None = None
    hsv_hi: list[int] | None = None
    min_area_px: float = 150.0
    max_area_frac: float = 0.15                # blobs bigger than this share of the image are background
    # Image direction the jaws close along (deg, 0 = image right). Lets long objects be gripped across.
    jaw_axis_deg: float = 0.0
    # Servo calibration (from `c` in the track tool). None until calibrated.
    target_px: list[float] | None = None      # where the object sits in the image when it's under the jaws
    jacobian_tool: list[list[float]] | None = None  # px per metre of tool-frame x/y motion (2x2)
    hover_m: float | None = None              # tool-point height above the table during calibration
    image_size: list[int] | None = None       # [width, height] of the calibration frames
    # Area the automatic search sweeps (base frame, m): in front of the arm, where it can point down.
    search_x_m: list[float] = field(default_factory=lambda: [0.17, 0.36])
    search_y_m: list[float] = field(default_factory=lambda: [-0.16, 0.16])

    @property
    def calibrated(self) -> bool:
        return self.target_px is not None and self.jacobian_tool is not None and self.hover_m is not None

    def save(self) -> None:
        data = {
            "camera_index": self.camera_index,
            "hsv_lo": self.hsv_lo, "hsv_hi": self.hsv_hi, "min_area_px": self.min_area_px,
            "max_area_frac": self.max_area_frac, "jaw_axis_deg": self.jaw_axis_deg,
            "target_px": self.target_px, "jacobian_tool": self.jacobian_tool, "hover_m": self.hover_m,
            "image_size": self.image_size, "search_x_m": self.search_x_m, "search_y_m": self.search_y_m,
        }
        header = "# Wrist-camera settings, written by `python -m scrub_arm track`.\n"
        _atomic_write(self.path, header + yaml.safe_dump(data, sort_keys=False, default_flow_style=None))


def load_vision(path: Path | str | None = None) -> VisionCfg:
    path = Path(path) if path else DEFAULT_VISION_PATH
    cfg = VisionCfg(path=path)
    if path.exists():
        d = yaml.safe_load(path.read_text()) or {}
        for k in ("camera_index", "hsv_lo", "hsv_hi", "min_area_px", "max_area_frac", "jaw_axis_deg",
                  "target_px", "jacobian_tool", "hover_m",
                  "image_size", "search_x_m", "search_y_m"):
            if d.get(k) is not None:
                setattr(cfg, k, d[k])
    return cfg


# ── detection ─────────────────────────────────────────────────────────────

GREY_SATURATION = 50  # below this a colour is grey/black/white and hue means nothing


def learn_color(frame_bgr: np.ndarray, u: int, v: int, radius: int = 6) -> tuple[list[int], list[int], str]:
    """HSV range for the object at pixel (u, v). Returns (lo, hi, note).

    Coloured objects are matched on hue with a saturation floor, so greys, whites and
    glare never match. Grey, black or white objects can't be told apart by hue, so they're
    matched on brightness and low saturation instead, which only works against a surface
    of a clearly different brightness.
    """
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    patch = hsv[max(0, v - radius):min(h, v + radius + 1), max(0, u - radius):min(w, u + radius + 1)]
    hue, sat, val = np.median(patch.reshape(-1, 3), axis=0)
    if sat >= GREY_SATURATION:
        lo = [int(hue - 10), int(max(GREY_SATURATION - 10, sat - 60)), int(max(30, val - 70))]
        hi = [int(hue + 10), 255, int(min(255, val + 70))]
        return lo, hi, "coloured object"
    lo = [0, 0, int(max(0, val - 45))]
    hi = [179, int(min(255, sat + 35)), int(min(255, val + 45))]
    tone = "dark" if val < 90 else "light" if val > 170 else "grey"
    return lo, hi, f"{tone} object with little colour: tracking by brightness, so use a contrasting surface"


def color_mask(frame_bgr: np.ndarray, lo: list[int], hi: list[int]) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Hue wraps at 180 (reds), so a learned range can spill past either end.
    h_lo, h_hi = lo[0], hi[0]
    if h_lo < 0:
        hue_ranges = [(0, h_hi), (h_lo + 180, 179)]
    elif h_hi > 179:
        hue_ranges = [(h_lo, 179), (0, h_hi - 180)]
    else:
        hue_ranges = [(h_lo, h_hi)]
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for a, b in hue_ranges:
        mask |= cv2.inRange(hsv, np.array([a, lo[1], lo[2]]), np.array([b, hi[1], hi[2]]))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def detect(frame_bgr: np.ndarray, cfg: VisionCfg) -> tuple[Detection | None, np.ndarray | None]:
    """Largest blob of the learned colour. Returns (detection or None, mask)."""
    import cv2

    if cfg.hsv_lo is None or cfg.hsv_hi is None:
        return None, None
    mask = color_mask(frame_bgr, cfg.hsv_lo, cfg.hsv_hi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = cfg.max_area_frac * mask.shape[0] * mask.shape[1]
    best, best_area = None, cfg.min_area_px
    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if best_area < area <= max_area:  # bigger than the object could be: wall, table, glare
            best, best_area = hull, area
    if best is None:
        return None, mask
    m = cv2.moments(best)
    (_, _), (rw, rh), rect_angle = cv2.minAreaRect(best)
    angle = rect_angle if rw >= rh else rect_angle + 90.0
    aspect = max(rw, rh) / max(min(rw, rh), 1.0)
    return Detection(center=(m["m10"] / m["m00"], m["m01"] / m["m00"]), area=best_area, hull=best,
                     angle_deg=float(angle % 180.0), aspect=float(aspect)), mask


def match_fraction(frame_bgr: np.ndarray, cfg: VisionCfg) -> float:
    """Share of the image the learned colour matches; large means it isn't distinctive."""
    mask = color_mask(frame_bgr, cfg.hsv_lo, cfg.hsv_hi)
    return float(np.count_nonzero(mask)) / mask.size


# ── cameras ───────────────────────────────────────────────────────────────

class Camera:
    """OpenCV camera read on a background thread; `latest()` returns the newest frame."""

    def __init__(self, index: int) -> None:
        self.index = index
        self._cap = None
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        import cv2

        self._cap = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION) if _is_mac() else cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"can't open camera {self.index}. On macOS: System Settings -> Privacy & Security -> Camera, "
                "allow your terminal / VS Code, then restart it. Try other indices with `--camera N`."
            )
        self._running = True
        threading.Thread(target=self._run, name="camera", daemon=True).start()
        deadline = time.monotonic() + 3.0
        while self.latest() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        if self.latest() is None:
            raise RuntimeError(f"camera {self.index} opened but sent no frames")

    def _run(self) -> None:
        while self._running:
            ok, f = self._cap.read()
            if ok:
                with self._lock:
                    self._frame, self._stamp = f, time.monotonic()
            else:
                time.sleep(0.01)

    def latest(self, newer_than: float = 0.0) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None and self._stamp > newer_than else None

    def fresh(self, timeout: float = 1.0) -> np.ndarray:
        """A frame captured after this call (so it reflects where the arm is now)."""
        t = time.monotonic()
        deadline = t + timeout
        while time.monotonic() < deadline:
            f = self.latest(newer_than=t)
            if f is not None:
                return f
            time.sleep(0.005)
        raise RuntimeError("camera stopped sending frames")

    def stop(self) -> None:
        self._running = False
        if self._cap is not None:
            self._cap.release()


def _is_mac() -> bool:
    import sys

    return sys.platform == "darwin"


class SimCamera:
    """Renders an orange disc on a grey table as seen from a down-facing wrist camera.

    The camera sits `back_m` up the gripper from the tool point and `side_m` off to one
    side (like the real bracket), looking along the gripper; the object is at world (x, y)
    on the table plane z = table_z.
    """

    def __init__(self, driver, kin, table_z: float, obj_xy: tuple[float, float],
                 back_m: float = 0.08, side_m: float = 0.03, f_px: float = 500.0, size=(480, 640)) -> None:
        self.driver, self.kin = driver, kin
        self.table_z, self.obj_xy = table_z, np.array(obj_xy, dtype=float)
        self.back_m, self.side_m, self.f, self.size = back_m, side_m, f_px, size
        self.radius_m = 0.02
        # Set to (length, width) in m for a long object like a clamp, lying at `obj_yaw` (rad, base frame).
        self.obj_size: tuple[float, float] | None = None
        self.obj_yaw = 0.0

    def image_axes(self) -> tuple[np.ndarray, np.ndarray]:
        """World directions of image right (+u) and image down (+v) for the current pose."""
        q = self.driver.q_measured()
        R, a = self.kin.fk(q)[:3, :3], self.kin.approach(q)
        side = R[:, int(np.argmin(np.abs(R.T @ a)))]               # a tool axis across the gripper
        x_img = side - np.dot(side, a) * a                          # image "right" fixed to the tool
        x_img /= np.linalg.norm(x_img)
        return x_img, np.cross(a, x_img)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def latest(self, newer_than: float = 0.0) -> np.ndarray:
        return self.render()

    def fresh(self, timeout: float = 1.0) -> np.ndarray:
        return self.render()

    def render(self) -> np.ndarray:
        import cv2

        h, w = self.size
        img = np.full((h, w, 3), (120, 120, 120), np.uint8)
        q = self.driver.q_measured()
        p, a = self.kin.fk(q)[:3, 3], self.kin.approach(q)        # a = optical axis
        x_img, y_img = self.image_axes()
        cam = p - self.back_m * a + self.side_m * x_img

        def project(world_xy):
            P = np.array([*world_xy, self.table_z]) - cam
            depth = float(np.dot(P, a))
            if depth <= 0.01:
                return None, depth
            return (w / 2 + self.f * np.dot(P, x_img) / depth, h / 2 + self.f * np.dot(P, y_img) / depth), depth

        if self.obj_size is None:
            c, depth = project(self.obj_xy)
            if c is None:
                return img
            r = max(2, int(self.f * self.radius_m / depth))
            if -r < c[0] < w + r and -r < c[1] < h + r:
                cv2.circle(img, (int(c[0]), int(c[1])), r, (30, 120, 230), -1)  # orange (BGR)
            return img
        L, W = self.obj_size
        along = np.array([np.cos(self.obj_yaw), np.sin(self.obj_yaw)])
        across = np.array([-along[1], along[0]])
        corners = [project(self.obj_xy + sl * L / 2 * along + sw * W / 2 * across)[0]
                   for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        if all(c is not None for c in corners):
            cv2.fillPoly(img, [np.array(corners, dtype=np.int32)], (30, 120, 230))
        return img


def tool_yaw(kin, q: np.ndarray) -> float:
    """Heading (rad) of the tool frame about vertical, so camera calibration can follow wrist rotation."""
    R = kin.fk(q)[:3, :3]
    a = kin.approach(q)
    for i in range(3):
        v = R[:, i]
        if abs(np.dot(v, a)) < 0.5:
            v = v - np.dot(v, [0, 0, 1]) * np.array([0, 0, 1.0])
            if np.linalg.norm(v) > 1e-6:
                return math.atan2(v[1], v[0])
    return 0.0
