"""Run the real pick motion onto empty table; photograph the fingertips open and closed at grasp height."""
import sys, time, json
from pathlib import Path
import numpy as np, cv2
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/arm")
from scrub_arm.app import build
from scrub_arm.teach import park_and_shutdown
from scrub_arm.handoff import TaskFailed
X, Y, YAW = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
out = Path(R + "/hw_check/real/gcal"); out.mkdir(parents=True, exist_ok=True)
app = build(); d = app.driver; a = app.assistant
d.start(); d.enable()
cap = cv2.VideoCapture(0); cap.set(3, 1920); cap.set(4, 1080)
def snap(name):
    time.sleep(0.5)
    for _ in range(6): cap.read()
    ok, f = cap.read(); cv2.imwrite(str(out / f"{name}.jpg"), f)
    json.dump({"q": d.q_measured().tolist(), "target": [X, Y, YAW]}, open(out / f"{name}.json", "w"))
_grasp = d.gripper_grasp
def look(close_mm=None):
    snap(f"{tag}_open"); ok = _grasp(close_mm); snap(f"{tag}_closed"); return ok
d.gripper_grasp = look
tag = f"x{X:.0f}_y{Y:.0f}_yaw{YAW:.0f}"
try:
    d.move_pose("scan", speed=0.5)
    try:
        a._pick_at(X, Y, YAW, 4.5, jaw_mm=(33.0, 1.0))
    except TaskFailed as e:
        print("expected miss:", e)
    d.move_pose("scan", speed=0.5)
finally:
    cap.release(); park_and_shutdown(app)
