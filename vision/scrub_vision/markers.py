"""ArUco fiducials on the tray: used to build the reach map and to catch camera-pose drift.

    python -m scrub_vision.markers --sheet data/markers.png          # printable sheet, ids 0..11
    python -m scrub_vision.markers --camera 0                        # which markers does the camera see?

Print the sheet at 100% scale and measure one marker's side: that is `marker_mm` in arm.yaml.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def detect(bgr: np.ndarray) -> dict[int, dict]:
    """{id: {"center": [u, v], "side_px": mean edge length}}"""
    corners, ids, _ = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters()).detectMarkers(bgr)
    out = {}
    for c, i in zip(corners, ids.flatten() if ids is not None else []):
        pts = c.reshape(4, 2)
        side = float(np.mean([np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]))
        out[int(i)] = {"center": [float(pts[:, 0].mean()), float(pts[:, 1].mean())], "side_px": side}
    return out


def without_markers(dets: list, bgr: np.ndarray) -> list:
    """Classical blobs include the printed tray markers; drop any blob sitting on one."""
    ms = detect(bgr).values()
    return [d for d in dets if all(np.hypot(d.center_px[0] - m["center"][0], d.center_px[1] - m["center"][1])
                                   > m["side_px"] for m in ms)]


def sheet(path: str, n: int = 12, cols: int = 4, px: int = 300, gap: int = 80) -> None:
    rows = -(-n // cols)
    img = np.full((rows * (px + gap) + gap, cols * (px + gap) + gap), 255, np.uint8)
    for i in range(n):
        r, c = divmod(i, cols)
        y, x = gap + r * (px + gap), gap + c * (px + gap)
        img[y:y + px, x:x + px] = cv2.aruco.generateImageMarker(DICT, i, px)
        cv2.putText(img, str(i), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2)
    cv2.imwrite(path, img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet")
    ap.add_argument("--camera", type=int)
    args = ap.parse_args()
    if args.sheet:
        sheet(args.sheet)
        print(f"wrote {args.sheet}")
    elif args.camera is not None:
        from .camera import Camera
        cam = Camera(args.camera)
        print(detect(cam.frame()))
        cam.close()
    else:
        ap.error("give --sheet or --camera")


if __name__ == "__main__":
    main()
