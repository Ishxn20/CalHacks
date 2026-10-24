"""Find objects on the tray and say how to grip them.

Two stages, so it scales from 2 sandbox objects to the 15 real tools:
  identity     YOLO (optional): which tool is this, and roughly where
  orientation  classical OpenCV on the plain background: mask -> minAreaRect -> OBB

The OBB stage is class-agnostic, so new tools need no retraining.

Angle convention (image coords: x right, y down; positive = clockwise on screen):
  long_deg  direction of the long axis, in [-90, 90)
  grip_deg  direction the jaws close = the short axis, in [-90, 90)

    python -m scrub_vision.grasp --camera 0                  # live overlay, q to quit
    python -m scrub_vision.grasp --image data/scalpels.jpg   # one still
    python -m scrub_vision.grasp --camera 0 --weights runs/tools/weights/best.pt --name scalpel_11
"""
from __future__ import annotations

import argparse
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

MIN_AREA_PX = 400      # shortcut: fixed pixel floor, scale it if the camera moves much closer or further
MIN_CONTRAST = 25      # Lab distance from the background below which we call the frame empty
ROI_FILE = Path(__file__).resolve().parents[1] / "data" / "roi.txt"   # x,y,w,h of the plain mat, from --select-roi


@dataclass
class Detection:
    name: str | None
    conf: float
    center_px: tuple[float, float]
    long_deg: float
    grip_deg: float
    length_px: float
    width_px: float
    heavy_dir: tuple[float, float]   # unit vector from the centre toward the bulkier end (the handle)
    box_px: list[list[float]]        # 4 OBB corners

    def to_json(self) -> dict:
        return asdict(self)


def wrap180(a: float) -> float:
    """Fold an angle into [-90, 90): the jaws are symmetric, so a and a+180 are the same grip."""
    return (a + 90.0) % 180.0 - 90.0


def load_roi() -> tuple[int, int, int, int] | None:
    return tuple(int(v) for v in ROI_FILE.read_text().split(",")) if ROI_FILE.exists() else None


def segment_in(bgr: np.ndarray, roi) -> list[Detection]:
    """`segment` inside the plain mat only. Without it the frame border (the table) is taken as the background."""
    if roi is None:
        return segment(bgr)
    x, y, w, h = roi
    return [_shift(d, x, y, d.name, d.conf) for d in segment(bgr[y:y + h, x:x + w], bg_from_all=True)]


def segment(bgr: np.ndarray, bg_from_all: bool = False) -> list[Detection]:
    """Every blob that differs from the (plain) background, as unnamed detections.

    shortcut: background = median colour of the image border (or of every pixel with `bg_from_all`, which
    survives a box that spills past the mat as long as objects fill under half of it), so lighting must be
    roughly flat. Upgrade: a per-pixel background model.
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    edge = lab.reshape(-1, 3) if bg_from_all else np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    dist = np.linalg.norm(lab - np.median(edge, axis=0), axis=2)
    if dist.max() < MIN_CONTRAST:
        return []
    d8 = np.clip(dist, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (d8 > max(t, MIN_CONTRAST)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    h, w = mask.shape
    out = []
    for cnt in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) < MIN_AREA_PX or x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue  # speck, or cut off by the frame edge (an OBB of half an object is a bad grasp)
        out.append(_detection(cnt))
    return sorted(out, key=lambda d: -d.length_px * d.width_px)


def _detection(cnt: np.ndarray) -> Detection:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    long_v = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    length, width = sorted((float(np.linalg.norm(e1)), float(np.linalg.norm(e2))), reverse=True)
    u = long_v / max(np.linalg.norm(long_v), 1e-9)
    long_deg = wrap180(math.degrees(math.atan2(u[1], u[0])))
    cx, cy = rect[0]
    filled = np.zeros((int(box[:, 1].max()) + 2, int(box[:, 0].max()) + 2), np.uint8)
    cv2.drawContours(filled, [cnt], -1, 255, -1)
    ys, xs = np.nonzero(filled)
    proj = (xs - cx) * u[0] + (ys - cy) * u[1]
    s = 1.0 if (proj > 0).sum() >= (proj < 0).sum() else -1.0
    return Detection(
        name=None, conf=1.0, center_px=(float(cx), float(cy)), long_deg=long_deg,
        grip_deg=wrap180(long_deg + 90.0), length_px=length, width_px=width,
        heavy_dir=(float(u[0] * s), float(u[1] * s)), box_px=box.tolist(),
    )


def refine(bgr: np.ndarray, xyxy, name: str, conf: float, pad: float = 0.15) -> Detection | None:
    """OBB for a YOLO box: segment a padded crop, take the blob under the box centre."""
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    h, w = bgr.shape[:2]
    ox, oy = max(int(x1 - px), 0), max(int(y1 - py), 0)
    crop = bgr[oy:min(int(y2 + py) + 1, h), ox:min(int(x2 + px) + 1, w)]
    if crop.size == 0:
        return None
    cx, cy = (x1 + x2) / 2 - ox, (y1 + y2) / 2 - oy
    cands = segment(crop)
    # Blob containing the box centre, else the biggest blob that overlaps the box.
    for d in cands:
        if cv2.pointPolygonTest(np.array(d.box_px, np.float32), (cx, cy), False) >= 0:
            return _shift(d, ox, oy, name, conf)
    return _shift(cands[0], ox, oy, name, conf) if cands else None


def _shift(d: Detection, ox: int, oy: int, name: str | None, conf: float) -> Detection:
    d.center_px = (d.center_px[0] + ox, d.center_px[1] + oy)
    d.box_px = [[x + ox, y + oy] for x, y in d.box_px]
    d.name, d.conf = name, conf
    return d


class Finder:
    """`weights=None` is classical-only: unnamed blobs, so identity is unchecked."""

    def __init__(self, weights: str | None = None, conf: float = 0.4, imgsz: int = 512, roi=None) -> None:
        self.conf, self.imgsz = conf, imgsz
        self._roi = roi   # None = re-read the saved file on every call, so `--select-roi` needs no restart
        self.model = None
        if weights:
            from ultralytics import YOLO
            self.model = YOLO(weights)

    def find(self, bgr: np.ndarray) -> list[Detection]:
        if self.model is None:
            return segment_in(bgr, self._roi or load_roi())
        res = self.model(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        for xyxy, cls, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            d = refine(bgr, xyxy, res.names[int(cls)], c)
            if d:
                out.append(d)
        return sorted(out, key=lambda d: -d.conf)


def pick_target(dets: list[Detection], names: list[str] | None) -> Detection | None:
    """Best detection for any of `names`. Unnamed (classical-only) blobs are taken as-is, biggest first."""
    ok = [d for d in dets if not names or d.name is None or d.name in names]
    return ok[0] if ok else None


def combine(dets: list[Detection]) -> tuple[Detection, float, float]:
    """Average detections of one object across frames -> (detection, centre spread px, angle spread deg)."""
    pts = np.array([d.center_px for d in dets])
    med = np.median(pts, axis=0)
    spread_px = float(np.max(np.linalg.norm(pts - med, axis=1)))
    z = np.exp(2j * np.radians([d.long_deg for d in dets]))      # angles repeat every 180 deg
    mean = z.mean()
    long_deg = wrap180(math.degrees(np.angle(mean)) / 2)
    spread_deg = float(max(abs(wrap180(d.long_deg - long_deg)) for d in dets))
    best = max(dets, key=lambda d: d.conf)
    out = Detection(
        name=best.name, conf=float(np.mean([d.conf for d in dets])), center_px=(float(med[0]), float(med[1])),
        long_deg=long_deg, grip_deg=wrap180(long_deg + 90.0), length_px=float(np.median([d.length_px for d in dets])),
        width_px=float(np.median([d.width_px for d in dets])), heavy_dir=best.heavy_dir, box_px=best.box_px,
    )
    return out, spread_px, spread_deg


def draw(frame: np.ndarray, dets: list[Detection], target: Detection | None = None) -> np.ndarray:
    img = frame.copy()
    for d in dets:
        col = (0, 255, 0) if d is target else (0, 180, 255)
        cv2.polylines(img, [np.int32(d.box_px)], True, col, 2)
        c = np.array(d.center_px)
        g = np.array([math.cos(math.radians(d.grip_deg)), math.sin(math.radians(d.grip_deg))]) * d.width_px * 0.8
        cv2.line(img, tuple(np.int32(c - g)), tuple(np.int32(c + g)), (0, 0, 255), 3)   # where the jaws go
        cv2.circle(img, tuple(np.int32(c)), 4, col, -1)
        cv2.putText(img, f"{d.name or 'object'} grip {d.grip_deg:+.0f}", (int(c[0]) + 8, int(c[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int)
    src.add_argument("--image")
    ap.add_argument("--weights", help="YOLO weights for identity; omit for classical-only")
    ap.add_argument("--name", help="highlight this class")
    ap.add_argument("--out", help="with --image: save the overlay here instead of opening a window")
    ap.add_argument("--select-roi", action="store_true",
                    help="with --camera: drag a box around the plain mat, Enter to save it for every tool")
    args = ap.parse_args()
    finder = Finder(args.weights)
    want = [args.name] if args.name else None

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"cannot read {args.image}")
        dets = finder.find(frame)
        for d in dets:
            print(f"{d.name or 'object'}: centre {d.center_px[0]:.0f},{d.center_px[1]:.0f} grip {d.grip_deg:+.1f} deg "
                  f"width {d.width_px:.0f}px length {d.length_px:.0f}px")
        img = draw(frame, dets, pick_target(dets, want))
        if args.out:
            cv2.imwrite(args.out, img)
        else:
            cv2.imshow("grasp", img)
            cv2.waitKey(0)
        return

    from .camera import Camera
    cam = Camera(args.camera)
    if args.select_roi:
        r = cv2.selectROI("drag a box around the plain mat (arm at tray_view), Enter to save", cam.frame())
        cam.close()
        cv2.destroyAllWindows()
        if r[2] > 0 and r[3] > 0:
            ROI_FILE.parent.mkdir(exist_ok=True)
            ROI_FILE.write_text(",".join(str(int(v)) for v in r))
            print(f"saved {r} -> {ROI_FILE}")
        return
    history: deque[Detection] = deque(maxlen=5)
    try:
        while True:
            frame = cam.frame()
            dets = finder.find(frame)
            target = pick_target(dets, want)
            # Smooth only what we show; the arm service does its own multi-frame averaging.
            if target:
                history.append(target)
                if len(history) >= 3 and max(math.dist(h.center_px, target.center_px) for h in history) < 15:
                    dets = [d for d in dets if d is not target]
                    target, *_ = combine(list(history))
                    dets.append(target)
            else:
                history.clear()
            cv2.imshow("grasp (q to quit)", draw(frame, dets, target))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
