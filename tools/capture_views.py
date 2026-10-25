"""Photograph the mat from many poses around the scan pose (camera keeps looking down). Saves jpg + json(q).
usage: capture_views.py OUTDIR [seed] [n]"""
import sys, json, time, itertools, random, math
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "arm"))
from scrub_arm.app import build
from scrub_arm.kinematics import Kinematics
from scrub_arm.teach import park_and_shutdown

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
rng = random.Random(seed)
app = build(); d = app.driver; kin = Kinematics(); lims = (app.cfg.lower, app.cfg.upper)
TABLE_Z = 0.002

def Rz(a):
    c, s = math.cos(a), math.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

d.start(); d.enable()
cap = cv2.VideoCapture(0); cap.set(3, 1920); cap.set(4, 1080)
try:
    d.move_pose("scan", speed=0.5); d.gripper_open()
    q_scan = d.pose_q("scan"); p0, R0 = kin.fk(q_scan)
    print("scan tool at", np.round(p0 * 1000), flush=True)
    views = [(0, 0, 0, 0)] + [(rng.uniform(-70, 70), rng.uniform(-90, 90), rng.uniform(-90, 25), rng.uniform(-35, 35))
                             for _ in range(N)]
    for i, (dx, dy, dz, yaw) in enumerate(views):
        p = p0 + np.array([dx, dy, dz]) / 1000
        if p[2] < TABLE_Z + 0.15:
            continue
        q = kin.ik_pose(p, Rz(math.radians(yaw)) @ R0, d.q_command(), lims)
        if q is None:
            q = kin.ik_pose(p, Rz(math.radians(yaw)) @ R0, q_scan, lims)
        if q is None or np.abs(q - d.q_command()).max() > math.radians(70):
            print(i, "skip (ik)", flush=True); continue
        try:
            d.check_limits(q); d.clear_abort(); d.move_to(q, speed=0.4)
        except Exception as e:  # noqa: BLE001
            print(i, "skip", e, flush=True); d.clear_fault(); continue
        time.sleep(0.6)
        for _ in range(6): cap.read()
        ok, f = cap.read()
        if not ok: continue
        stem = out / f"v{seed:02d}_{i:03d}"
        cv2.imwrite(str(stem) + ".jpg", f)
        json.dump({"q": d.q_measured().tolist(), "view": [dx, dy, dz, yaw]}, open(str(stem) + ".json", "w"))
        print(i, "saved", [round(v) for v in (dx, dy, dz, yaw)], flush=True)
    d.move_pose("scan", speed=0.5)
finally:
    cap.release(); park_and_shutdown(app)
