"""Rebuild perception/config/finger_mask.png from views taken at many arm poses with the gripper open.

    DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/capture_views.py hw_check/real/maskviews2 12 24
    .venv/bin/python tools/build_finger_mask.py hw_check/real/maskviews2

The camera rides on the gripper, so the fingers sit at the same pixels in every view while the table moves: a pixel
whose colour hardly changes across the views is finger. Keeps the steady regions that touch the bottom edge (where the
fingers enter the frame), grows them by a margin, backs up the old mask and writes a preview with both outlined.
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

R = Path(__file__).resolve().parents[1]
OUT = R / "perception/config/finger_mask.png"
SCALE = 4             # work at 1/4 resolution
STEADY = 10.0         # max per-pixel std (Lab units) across views to count as finger
MARGIN_PX = 8         # full-res growth so the finger edge never reads as a part


def main() -> None:
    views = sorted(Path(sys.argv[1]).glob("*.jpg"))
    if len(views) < 8:
        sys.exit(f"need at least 8 views, found {len(views)}")
    imgs = [cv2.cvtColor(cv2.imread(str(v)), cv2.COLOR_BGR2LAB) for v in views]
    h, w = imgs[0].shape[:2]
    small = np.stack([cv2.resize(i, (w // SCALE, h // SCALE), interpolation=cv2.INTER_AREA) for i in imgs]).astype(np.float32)
    sd = small.std(0).mean(-1)                                    # per-pixel colour spread across the views
    steady = (sd < STEADY).astype(np.uint8)
    steady = cv2.morphologyEx(steady, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(steady)
    keep = np.zeros_like(steady)
    for i in range(1, n):
        top, hh = st[i, cv2.CC_STAT_TOP], st[i, cv2.CC_STAT_HEIGHT]
        if top + hh >= steady.shape[0] - 1 and st[i, cv2.CC_STAT_AREA] > 50:     # touches the bottom edge
            keep[lab == i] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.resize(keep, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = cv2.dilate(mask, np.ones((2 * MARGIN_PX + 1,) * 2, np.uint8)) * 255
    old = cv2.imread(str(OUT), 0)
    if old is not None:
        shutil.copy(OUT, OUT.with_name("finger_mask_prev.png"))
    cv2.imwrite(str(OUT), mask)
    prev = cv2.cvtColor(imgs[0], cv2.COLOR_LAB2BGR)
    for m, col in ((old, (0, 0, 255)), (mask, (0, 255, 0))):
        if m is not None:
            cs, _ = cv2.findContours((cv2.resize(m, (w, h)) > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(prev, cs, -1, col, 3)
    cv2.imwrite(str(Path(sys.argv[1]) / "finger_mask_preview.jpg"), prev)
    print(f"{len(views)} views: finger mask covers {(mask > 0).mean():.1%} of the frame "
          f"(was {(old > 0).mean():.1%})" if old is not None else "", f"-> {OUT.relative_to(R)}; "
          f"preview (green new, red old): {Path(sys.argv[1]) / 'finger_mask_preview.jpg'}")


if __name__ == "__main__":
    main()
