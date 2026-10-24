"""Background webcam reader that always holds the latest undistorted frame."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int, K: np.ndarray | None, dist: np.ndarray | None,
                 width: int | None = None, height: int | None = None) -> None:
        self.cap = cv2.VideoCapture(index)
        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.K, self.dist = K, dist
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._maps = None
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if self.K is not None:
                if self._maps is None:
                    h, w = f.shape[:2]
                    self._maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (w, h), cv2.CV_16SC2)
                f = cv2.remap(f, *self._maps, cv2.INTER_LINEAR)
            with self._lock:
                self._frame = f

    def latest(self, timeout_s: float = 3.0) -> np.ndarray:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy()
            time.sleep(0.02)
        raise RuntimeError("camera produced no frames")
