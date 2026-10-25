"""At scan (nothing in the jaws): set each tool's open width, photograph, measure the real gap with the camera."""
import sys, time, json
from pathlib import Path
import numpy as np, cv2, yaml
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/arm")
from scrub_arm.app import build
from scrub_arm.teach import park_and_shutdown
from scrub_arm.kinematics import Kinematics
cam = yaml.safe_load(open(R + "/config/camera.yaml")); K = np.array(cam["K"]); D = np.array(cam["dist"])
pt = np.array(cam["handeye"]["p_tool_cam_m"]); Rt = np.array(cam["handeye"]["R_tool_cam"]); jaw = Kinematics().jaw_axis_local
proj = lambda p: cv2.projectPoints((Rt.T @ (p - pt)).reshape(1, 3), np.zeros(3), np.zeros(3), K, D)[0].ravel()
def gap(img):
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    def first(sign):
        for w in np.arange(0, 140, 0.5):
            u, v = proj(sign * jaw * w / 2000).astype(int)
            if L[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3].mean() < 45: return w
    return (first(1) + first(-1)) / 2 - 13.75          # camera sees the finger bodies: 13.75 mm at fully closed
app = build(); d = app.driver
d.start(); d.enable()
cap = cv2.VideoCapture(0); cap.set(3, 1920); cap.set(4, 1080)
try:
    d.move_pose("scan", speed=0.5)
    for iid, inst in app.tray.instruments.items():
        if not inst.jaw_mm: continue
        for kind, w in zip(("open", "close"), inst.jaw_mm):
            d.gripper_width(w, wait_s=1.2)
            for _ in range(6): cap.read()
            ok, f = cap.read()
            print(f"{iid:24s} {kind:5s} target {w:4.0f} mm  measured {gap(f):5.1f} mm", flush=True)
    d.gripper_open()
finally:
    cap.release(); park_and_shutdown(app)
