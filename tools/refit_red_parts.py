"""Refit red_parts.yaml classes (6 features) from the old views (old layout) + the new views (current layout,
incl. the open micro scissors), labelled by footprint centroid; robust stats (median, 1.4826 MAD)."""
import sys, json, glob
import numpy as np, cv2, yaml
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/perception")
from scrub_vision.red_parts import RedPartsDetector, MODEL_YAML
FLOOR = np.array([3.0, 0.6, 0.6, 0.8, 0.05, 0.05])
det = RedPartsDetector(); det.model = dict(det.model); det.model["classes"] = {}
det.model["grasp_offset_mm"] = [0.0, 0.0]

def views(d):
    for j in sorted(glob.glob(f"{R}/hw_check/real/{d}/*.json")):
        yield cv2.imread(j[:-5] + ".jpg"), np.array(json.load(open(j))["q"])

def complete(img, q):
    cv_ = det.canvas(q, img.shape); out = []
    for x in det.detect(img, q, grasp_offset=False):
        if x.exposure == 1:
            r, c = np.nonzero(x.table_mask); cen = cv_.to_mm(np.c_[r.mean(), c.mean()])[0]
            if 380 < cen[0] < 560:                                   # on the paper (not clutter on the desk)
                out.append((cen, x.shape_features))
    return out

samples = []
# old layout: LAYOUT centres (bbox centres ~ centroids)
OLD = {"scalpel_11_long": (470, 68), "needle_holder_ryder": (439, 23), "micro_scissors@side": (436, -10),
       "needle_holder_short": (419, -50), "scalpel_10": (447, -81), "forceps_fine_vascular": (466, -108),
       "micro_scissors@flat": (451, -143), "scalpel_20": (440, -187)}
for d in ("views", "test"):
    for img, q in views(d):
        for cen, f in complete(img, q):
            k, dist = min(((k, np.linalg.norm(cen - np.array(v))) for k, v in OLD.items()), key=lambda t: t[1])
            if dist < 25: samples.append((k, f))
# current layout: reference = a new view with all 8 complete, left to right (+y first)
NEW_ORDER = ["needle_holder_short", "micro_scissors@side", "needle_holder_ryder", "scalpel_10",
             "forceps_fine_vascular", "micro_scissors@open", "scalpel_20", "scalpel_11_long"]
newv = list(views("openviews"))
ref = next(sorted(c, key=lambda t: -t[0][1]) for c in (complete(i, q) for i, q in newv) if len(c) == 8)
REF = {n: c for n, (c, _) in zip(NEW_ORDER, ref)}
print("current layout:", {k: np.round(v).tolist() for k, v in REF.items()})
for img, q in newv:
    for cen, f in complete(img, q):
        k, dist = min(((k, np.linalg.norm(cen - v)) for k, v in REF.items()), key=lambda t: t[1])
        if dist < 25: samples.append((k, f))
cls = {}
for k in sorted({s[0] for s in samples}):
    X = np.array([f["vec"] for c, f in samples if c == k])
    med = np.median(X, 0); mad = 1.4826 * np.median(np.abs(X - med), 0)
    ringed = np.mean([f["holes"] >= 2 for c, f in samples if c == k]) > 0.5
    cls[k] = {"mu": med.round(3).tolist(), "sd": np.maximum(mad, FLOOR).round(3).tolist(), "ringed": bool(ringed), "n": len(X)}
    print(f"{k:24s} n {len(X):3d} mu {med.round(2)} ringed {ringed}")
old = yaml.safe_load(open(MODEL_YAML))
text = ("# Fitted 2026-10-24 (6 features: length, mean width, median width, narrow-end width (mm), narrow end / median\n"
        "# width, solidity) on real wrist-camera views of two layouts, robust stats (median, 1.4826 MAD), by\n"
        "# tools/refit_red_parts.py; includes the open micro scissors (blades spread into a loop).\n"
        + yaml.safe_dump({"classes": cls})
        + "# Measured joints (rad) at the scan pose with the gripper open: the camera pose the service assumes.\n"
        + f"scan_q: {json.dumps(old['scan_q'])}\n"
        + "# Added to every grab point: where the closed fingertips meet at grasp height (-5.0, -3.4) mm, + (0, +7.5)\n"
        + "# because the jaws still came down 5-10 mm right of the tool in the camera view (-y; user, kept 2026-10-24).\n"
        + f"grasp_offset_mm: {json.dumps(old['grasp_offset_mm'])}\n")
open(MODEL_YAML, "w").write(text)
print("written", MODEL_YAML)
