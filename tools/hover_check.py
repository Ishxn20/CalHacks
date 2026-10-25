"""Scan -> detect -> hover the open gripper 30 mm above each part's top at its grab point (no contact) -> photo."""
import sys, time, math, json
from pathlib import Path
import numpy as np, cv2
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/arm"); sys.path.insert(0, R + "/perception")
from scrub_arm.app import build
from scrub_arm.teach import park_and_shutdown
from scrub_vision.red_parts import RedPartsDetector
OFF = np.array([float(sys.argv[2]), float(sys.argv[3])]) if len(sys.argv) > 3 else np.zeros(2)
out = Path(R + "/hw_check/real/" + sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
app = build(); d = app.driver; kin = app.assistant.kin
det = RedPartsDetector(); table_z = 0.0021
d.start(); d.enable()
cap = cv2.VideoCapture(0); cap.set(3, 1920); cap.set(4, 1080)
def snap(name):
    time.sleep(0.6)
    for _ in range(6): cap.read()
    ok, f = cap.read(); cv2.imwrite(str(out / f"{name}.jpg"), f); return f
try:
    d.move_pose("scan", speed=0.5); d.gripper_open()
    f = snap("scan"); q = d.q_measured()
    dets = [x for x in det.detect(f, q) if x.fine_id or x.candidates]
    print("found", [(x.fine_id or "?" + x.candidates[0], np.round(x.grasp_mm)) for x in dets], flush=True)
    for x in dets:
        name = (x.fine_id or "maybe_" + x.candidates[0]) + f"_y{x.grasp_mm[1]:+.0f}"
        z = table_z + x.thickness_mm / 1000 + 0.03
        gx, gy = np.array(x.grasp_mm) + OFF
        qh = kin.ik_topdown(gx / 1000, gy / 1000, z, math.radians(x.yaw_deg), d.q_command(),
                            (app.cfg.lower, app.cfg.upper), allow_flip=True)
        if qh is None:
            print(name, "unreachable", flush=True); continue
        try:
            d.check_limits(qh); d.clear_abort(); d.move_pose("scan", speed=0.5); d.move_to(qh, speed=0.3, exact=True)
        except Exception as e:  # noqa: BLE001
            print(name, "skip", e, flush=True); d.clear_fault(); continue
        snap("hover_" + name)
        json.dump({"q": d.q_measured().tolist(), "grasp_mm": list(x.grasp_mm), "aimed_mm": [gx, gy], "yaw_deg": x.yaw_deg,
                   "axis_deg": x.axis_deg, "thickness_mm": x.thickness_mm}, open(out / f"hover_{name}.json", "w"))
        print(name, "hover photo", flush=True)
    d.move_pose("scan", speed=0.5)
finally:
    cap.release(); park_and_shutdown(app)
