"""Calibration targets without printed markers: click the centre of each physical target in the camera image.

    python -m scrub_vision.targets --camera 0     # arm at tray_view; click each target, u = undo, Enter = save, q = cancel

Targets are anything the jaws can be centred on by eye (packets, coins, tape). Spread them over the whole area
tools will lie in, not in a line; 4 is the minimum, 6 to 9 is better. The click order is the target id.
`serve` reports them from /markers when no printed markers are visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

TARGETS_FILE = Path(__file__).resolve().parents[1] / "data" / "targets.json"


def load_targets() -> dict[str, list[float]]:
    return json.loads(TARGETS_FILE.read_text()) if TARGETS_FILE.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, required=True)
    args = ap.parse_args()
    from .camera import Camera
    cam = Camera(args.camera)
    frame = cam.frame()
    cam.close()
    pts: list[list[float]] = []
    title = "click each target centre (u undo, Enter save, q cancel)"

    def show() -> None:
        img = frame.copy()
        for i, (u, v) in enumerate(pts):
            cv2.circle(img, (int(u), int(v)), 6, (0, 255, 0), -1)
            cv2.putText(img, str(i), (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(title, img)

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, lambda ev, x, y, *_: (pts.append([float(x), float(y)]), show()) if ev == cv2.EVENT_LBUTTONDOWN else None)
    show()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
            show()
        elif k in (13, 10):
            if len(pts) < 3:
                print("need at least 3 targets")
                continue
            TARGETS_FILE.parent.mkdir(exist_ok=True)
            TARGETS_FILE.write_text(json.dumps({str(i): p for i, p in enumerate(pts)}))
            print(f"saved {len(pts)} targets -> {TARGETS_FILE}")
            break
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
