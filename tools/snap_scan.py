"""Go to scan, open gripper, save a few frames, park at rest."""
import sys, time
from pathlib import Path
import cv2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "arm"))
from scrub_arm.app import build
from scrub_arm.teach import park_and_shutdown

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
app = build(); d = app.driver
d.start(); d.enable()
try:
    d.move_pose("scan", speed=0.5); d.gripper_open()
    cap = cv2.VideoCapture(0); cap.set(3, 1920); cap.set(4, 1080)
    for _ in range(15): cap.read()
    time.sleep(0.5)
    for i in range(3):
        ok, f = cap.read(); stem = out / f"scan_{int(time.time())}_{i}"; cv2.imwrite(str(stem) + ".jpg", f)
        import json; json.dump({"q": d.q_measured().tolist()}, open(str(stem) + ".json", "w"))
    cap.release()
    print("saved", f.shape)
finally:
    park_and_shutdown(app)
