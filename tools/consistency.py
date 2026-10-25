"""How consistently do different camera poses place the same (unmoved) parts? Per view: centroids of complete
parts (table mm), matched to the median layout; report the spread. Needs no board."""
import sys, json, glob
import numpy as np, cv2
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/perception")
from scrub_vision.red_parts import RedPartsDetector
det = RedPartsDetector()

def cents(d):
    out = []
    for j in sorted(glob.glob(f"{R}/hw_check/real/{d}/*.json")):
        img = cv2.imread(j[:-5] + ".jpg"); q = np.array(json.load(open(j))["q"])
        cv_ = det.canvas(q, img.shape); v = []
        for x in det.detect(img, q, grasp_offset=False):
            if x.exposure == 1:
                r, c = np.nonzero(x.table_mask); cen = cv_.to_mm(np.c_[r.mean(), c.mean()])[0]
                if 380 < cen[0] < 560: v.append(cen)
        out.append((j, q, np.array(v)))
    return out

for d in sys.argv[1:]:
    V = cents(d)
    allc = np.vstack([v for _, _, v in V if len(v)])
    # layout = cluster centres (greedy 20 mm)
    ref = []
    for c in allc:
        if all(np.linalg.norm(c - r) > 20 for r in ref): ref.append(c)
    ref = np.array(ref)
    for _ in range(3):
        lab = [np.argmin(np.linalg.norm(ref - c, axis=1)) for c in allc]
        ref = np.array([np.median(allc[np.array(lab) == k], 0) for k in range(len(ref))])
    res = np.array([c - ref[np.argmin(np.linalg.norm(ref - c, axis=1))] for c in allc])
    res = res[np.linalg.norm(res, axis=1) < 40]
    print(f"{d:12s} views {len(V):3d} parts {len(res):4d}  spread x {res[:,0].std():5.1f} mm  y {res[:,1].std():5.1f} mm")
