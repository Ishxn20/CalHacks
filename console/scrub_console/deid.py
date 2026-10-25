"""On-device de-identification: faces are blurred before a frame is written to disk or leaves the robot.

Uses OpenCV's YuNet face detector (opencv_zoo, MIT). The 230 KB model is downloaded once and checked
against a pinned SHA-256. If it can't be loaded, recording is refused (fail closed); video never
leaves the robot un-blurred. Detectors miss some faces (masks, caps, steep angles), so Claude also
flags privacy problems and flagged clips need a person before they enter the dataset.
"""
from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
CACHE = Path(os.environ.get("SCRUB_FLEET_CACHE", Path.home() / ".cache" / "scrub_fleet"))


class DeidUnavailable(RuntimeError):
    pass


def model_path() -> Path:
    p = CACHE / "face_detection_yunet_2023mar.onnx"
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        try:
            urllib.request.urlretrieve(MODEL_URL, tmp)
        except OSError as e:
            raise DeidUnavailable(f"could not download the face detector: {e}") from e
        tmp.rename(p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        p.unlink()
        raise DeidUnavailable(f"face detector checksum mismatch ({digest[:12]}...); deleted, will re-download")
    return p


class FaceBlur:
    def __init__(self, score: float = 0.6) -> None:
        self._det = cv2.FaceDetectorYN.create(str(model_path()), "", (320, 320), score, 0.3, 5000)
        self._lock = threading.Lock()

    def find(self, bgr: np.ndarray, max_side: int = 640) -> list[tuple[int, int, int, int]]:
        h, w = bgr.shape[:2]
        s = min(1.0, max_side / max(h, w))
        img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
        with self._lock:
            self._det.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self._det.detect(img)
        if faces is None:
            return []
        return [tuple(int(v / s) for v in f[:4]) for f in faces]

    def __call__(self, bgr: np.ndarray) -> int:
        """Blur every detected face in place (with a margin); returns how many were found."""
        boxes = self.find(bgr)
        h, w = bgr.shape[:2]
        for x, y, bw, bh in boxes:
            m = int(0.3 * max(bw, bh))
            x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(w, x + bw + m), min(h, y + bh + m)
            roi = bgr[y0:y1, x0:x1]
            if roi.size:
                k = max(31, (max(roi.shape[:2]) // 2) | 1)
                bgr[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
        return len(boxes)
