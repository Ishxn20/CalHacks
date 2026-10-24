"""See a camera.

    python -m scrub_vision.camera --probe        # which indices open; saves data/cam_<n>.jpg so you can see which is which
    python -m scrub_vision.camera --index 0      # live window, press q to quit
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path

import cv2

DATA = Path(__file__).resolve().parents[1] / "data"


class Camera:
    """Keeps only the newest frame (a thread drains the buffer), so `frame()` is never stale."""

    def __init__(self, index: int, warmup: int = 15) -> None:
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {index} (macOS: allow camera access for this terminal, then retry)")
        for _ in range(warmup):   # auto-exposure needs a few frames; the first ones are black
            self.cap.read()
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        ok, first = self.cap.read()
        if not ok:
            raise RuntimeError(f"camera {index} opened but gives no frames")
        self._frame = first
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        while self._running:
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame = f

    def frame(self):
        with self._lock:
            return self._frame.copy()

    def close(self) -> None:
        self._running = False
        self.cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()

    if args.probe:
        DATA.mkdir(exist_ok=True)
        for i in range(5):
            cap = cv2.VideoCapture(i)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            print(i, f"frame {frame.shape[1]}x{frame.shape[0]} -> {DATA / f'cam_{i}.jpg'}" if ok else "no frame")
            if ok:
                cv2.imwrite(str(DATA / f"cam_{i}.jpg"), frame)
            cap.release()
        return

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.index} (macOS: allow camera access for this terminal, then retry)")
    while True:
        ok, frame = cap.read()
        if ok:
            cv2.imshow(f"camera {args.index} (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
