"""Fit / test the red-parts identifier (scrub_vision/red_parts.py) on real wrist-camera views.

    python scripts/fit_red_parts.py fit  ../hw_check/real/views     -> config/red_parts.yaml
    python scripts/fit_red_parts.py test ../hw_check/real/test

Each view is <stem>.jpg + <stem>.json ({"q": joints in rad at the photo}), taken by hw_check capture runs.
Labels come from where each part lies on the table: the parts were not moved between views (LAYOUT: grab points
in table mm, measured from the scan view).
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from scrub_vision.red_parts import MODEL_YAML, RedPartsDetector, part_of

# The layout on the paper during capture (2026-10-24), left to right in the scan image. The two micro scissors
# rest differently (back-facing on its side, forward-facing flat), so they're separate classes of one part.
LAYOUT = {"scalpel_11_long": (470, 68), "needle_holder_ryder": (439, 23), "micro_scissors@side": (436, -10),
          "needle_holder_short": (419, -50), "scalpel_10": (447, -81), "forceps_fine_vascular": (466, -108),
          "micro_scissors@flat": (451, -143), "scalpel_20": (440, -187)}
MATCH_MM = 25.0
FLOOR = [3.0, 0.6, 0.6, 0.8, 0.05]        # measurement noise floor per feature (mm, mm, mm, mm, ratio)


def views(d: Path):
    for j in sorted(d.glob("*.json")):
        yield j.stem, cv2.imread(str(j.with_suffix(".jpg"))), np.array(json.loads(j.read_text())["q"])


def labelled(det: RedPartsDetector, d: Path):
    """[(view, class, features, detection)] for every complete part, labelled by position."""
    names, ref = list(LAYOUT), np.array(list(LAYOUT.values()), float)
    out = []
    for stem, bgr, q in views(d):
        for dd in det.detect(bgr, q):
            if dd.exposure < 1:
                continue
            k = int(np.argmin(np.linalg.norm(ref - np.array(dd.grasp_mm), axis=1)))
            if np.linalg.norm(ref[k] - dd.grasp_mm) < MATCH_MM:
                out.append((stem, names[k], dd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit", "test"])
    ap.add_argument("dir", type=Path)
    a = ap.parse_args()
    det = RedPartsDetector(model_yaml=MODEL_YAML) if a.mode == "test" else None
    if det is None:                                    # fitting: features only, no model yet
        MODEL_YAML.write_text(yaml.safe_dump({"classes": {}}))
        det = RedPartsDetector(model_yaml=MODEL_YAML)
    rows = labelled(det, a.dir)
    if a.mode == "fit":
        cls = {}
        for c in sorted({r[1] for r in rows}):
            X = np.array([r[2].shape_features["vec"] for r in rows if r[1] == c])
            ringed = np.mean([r[2].shape_features["holes"] >= 2 for r in rows if r[1] == c]) > 0.5
            cls[c] = {"mu": X.mean(0).round(3).tolist(), "sd": np.maximum(X.std(0), FLOOR).round(3).tolist(),
                      "ringed": bool(ringed), "n": len(X)}
        MODEL_YAML.write_text("# Fitted by scripts/fit_red_parts.py on " + str(a.dir) + f" ({len(rows)} parts, "
                              f"{len({r[0] for r in rows})} views).\n# features: length, mean width, median width, "
                              "narrow-end width (mm), narrow end / median width\n" + yaml.safe_dump({"classes": cls}))
        print(f"fitted {len(cls)} classes on {len(rows)} parts -> {MODEL_YAML}")
        return
    n_views = len({r[0] for r in rows})
    ok = sum(r[2].fine_id is not None and part_of(r[1]) in (r[2].fine_id, *r[2].alt_ids, _part(r[2])) for r in rows)
    wrong = [(r[0], part_of(r[1]), r[2].fine_id, r[2].candidates) for r in rows
             if not (r[2].fine_id is not None and part_of(r[1]) in (r[2].fine_id, *r[2].alt_ids, _part(r[2])))]
    print(f"{ok}/{len(rows)} = {ok / max(len(rows), 1):.3f} correct over {n_views} views")
    print("per part:", dict(Counter(part_of(r[1]) for r in rows)))
    for w in wrong:
        print("  wrong:", w)


def _part(d):
    return "micro_scissors" if d.fine_id in ("scissors_micro_forward", "scissors_micro_back") else d.fine_id


if __name__ == "__main__":
    main()
