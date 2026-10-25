"""Per hover photo: fingertip landing point (aim px at 3 cm) vs the target's centreline seen from the hover, mm."""
import sys, json, glob
import numpy as np, cv2, yaml
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/perception")
from scrub_vision.red_parts import Camera, red_mask, load_fingers
cam = Camera(); AIM = yaml.safe_load(open(R + "/config/camera.yaml"))["aim_px"][3]
errs = []
for j in sorted(glob.glob(R + f"/hw_check/real/{sys.argv[1]}/hover_*.json")):
    h = json.load(open(j)); q = np.array(h["q"]); z = h["thickness_mm"]
    img = cv2.imread(j[:-5] + ".jpg")
    m = red_mask(img) & ~load_fingers(img.shape)
    n, lab = cv2.connectedComponents(m.astype(np.uint8))
    tip = cam.px_to_mm([AIM], q, z)[0]
    best = None
    for i in range(1, n):
        vs, us = np.nonzero(lab == i)
        if len(us) < 2000: continue
        P = cam.px_to_mm(np.c_[us, vs][:: max(1, len(us) // 4000)], q, z)
        mu = P.mean(0); w, v = np.linalg.eigh(np.cov((P - mu).T)); ax = v[:, -1]
        # centreline near the fingertip: points within 15 mm along the axis of the tip's projection
        u = (P - mu) @ ax; ut = (tip - mu) @ ax
        near = P[np.abs(u - ut) < 15]
        if len(near) < 20: continue
        c = near.mean(0)
        dist = np.linalg.norm(c - tip)
        if best is None or dist < best[0]: best = (dist, c, ax)
    if best is None:
        print(j.split("/")[-1], "target not found"); continue
    d, c, ax = best
    across = np.array([-ax[1], ax[0]])
    e = c - tip                                      # where the part's centreline is, relative to the fingertips
    errs.append(e)
    print(f"{j.split('hover_')[-1][:-5]:26s} part centre - fingertips = ({e[0]:+5.1f}, {e[1]:+5.1f}) mm, across axis {e @ across:+5.1f}")
E = np.array(errs); print("mean", E.mean(0).round(1), "sd", E.std(0).round(1))
