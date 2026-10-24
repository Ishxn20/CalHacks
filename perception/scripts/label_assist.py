"""Pre-label real photos with the current model so a human only corrects them.

    python scripts/label_assist.py --weights weights/pile_seg.pt --images data/real/images
Writes YOLO-seg polygons to data/real/labels/<stem>.txt (import both folders into Roboflow / Label Studio,
fix, export as YOLO, then train on synth + real with scrub_vision.train).
"""
import argparse
from pathlib import Path

import numpy as np

from scrub_vision.service import YoloSegmenter
from scrub_vision.synth import write_yolo_labels
from scrub_vision.templates import classes

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--images", default="data/real/images")
args = ap.parse_args()
import cv2  # noqa: E402

seg, names = YoloSegmenter(args.weights, conf=0.25), classes()
img_dir = Path(args.images)
lab_dir = img_dir.parent / "labels"
lab_dir.mkdir(exist_ok=True)
for f in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
    bgr = cv2.imread(str(f))
    segs = seg(bgr, None)
    n = write_yolo_labels(lab_dir / f"{f.stem}.txt", bgr.shape[:2], [m for m, _, _ in segs],
                          [names.index(c) for _, c, _ in segs])   # photo left untouched
    print(f"{f.name}: {n} instruments")
print(f"labels in {lab_dir}; correct them, then train.")
