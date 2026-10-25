"""Draw each detected part's outline, target grab point and jaw line on a scan frame."""
import sys, json, glob
import numpy as np, cv2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "perception"))
from scrub_vision.red_parts import RedPartsDetector
stem = sys.argv[1]; out = sys.argv[2]
img = cv2.imread(stem + ".jpg"); q = np.array(json.load(open(stem + ".json"))["q"])
det = RedPartsDetector(); vis = img.copy()
for raw in (False, True):
    for x in det.detect(img, q, grasp_offset=raw):
        if not x.candidates: continue
        if raw:
            name = x.fine_id or "?" + "/".join(c.split("_")[-1] for c in x.candidates)
            cs, _ = cv2.findContours(x.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 200, 0) if x.exposure == 1 else (0, 165, 255), 2)
        g = np.array(x.grasp_mm); a = np.radians(x.yaw_deg); j = np.array([np.cos(a), np.sin(a)])
        half = (x.width_mm / 2 + 4)
        pts = det.cam.mm_to_px(np.array([g, g + half * j, g - half * j]), q).astype(int)
        col = (255, 255, 0) if raw else (255, 0, 255)       # cyan = where the arm aims, magenta = raw centre
        cv2.line(vis, tuple(pts[1]), tuple(pts[2]), col, 3); cv2.circle(vis, tuple(pts[0]), 7, col, -1)
        if raw:
            cv2.putText(vis, f"{name} w{x.width_mm:.0f}", (pts[0][0] - 70, pts[0][1] - 18), 0, 0.7, (0, 0, 0), 5)
            cv2.putText(vis, f"{name} w{x.width_mm:.0f}", (pts[0][0] - 70, pts[0][1] - 18), 0, 0.7, (255, 255, 255), 2)
            print(f"{name:28s} aim {np.round(g, 1)} yaw {x.yaw_deg:6.1f} width {x.width_mm:4.1f} exposure {x.exposure:.2f}")
cv2.imwrite(out, vis)
