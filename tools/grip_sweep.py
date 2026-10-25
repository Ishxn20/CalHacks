"""At the scan pose (nothing in the jaws), step the gripper through angles and photograph the fingertips."""
import sys, time, json, math
from pathlib import Path
import numpy as np, cv2
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/arm")
from scrub_arm.app import build
from scrub_arm.teach import park_and_shutdown
out = Path(R + "/hw_check/real/grip"); out.mkdir(parents=True, exist_ok=True)
app = build(); d = app.driver; g = app.cfg.gripper
d.start(); d.enable()
cap = cv2.VideoCapture(0); cap.set(3, 1920); cap.set(4, 1080)
try:
    d.move_pose("scan", speed=0.5)
    for frac in np.linspace(1.0, 0.0, 11):
        cmd = g.closed + frac * (g.open - g.closed)
        d.set_gripper(cmd); d.sleep(1.2)
        for _ in range(6): cap.read()
        ok, f = cap.read()
        meas = float(d.read_fresh()[-1])
        name = f"g{int(round(frac * 100)):03d}"
        cv2.imwrite(str(out / f"{name}.jpg"), f)
        json.dump({"cmd_rad": cmd, "meas_rad": meas, "q": d.q_measured().tolist()}, open(out / f"{name}.json", "w"))
        print(name, round(math.degrees(cmd), 1), round(math.degrees(meas), 1), flush=True)
    d.gripper_open()
finally:
    cap.release(); park_and_shutdown(app)
