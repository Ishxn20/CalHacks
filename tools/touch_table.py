"""Open jaws, pick orientation, step down 1 mm at a time onto empty table until contact; report FK z."""
import sys, math, time
import numpy as np
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/arm")
from scrub_arm.app import build
from scrub_arm.teach import park_and_shutdown
X, Y, YAW = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
app = build(); d = app.driver; a = app.assistant; pk = app.cfg.pick
pk.descend_step = 0.001
d.start(); d.enable()
try:
    d.move_pose("scan", speed=0.5)
    zh = pk.table_z + pk.hover
    tilt = a._tilt_for(X, Y, YAW, pk.table_z, zh)
    d.move_to(a._solve(X, Y, zh, YAW, True, tilt), exact=True)
    d.gripper_width(33.0)
    z = a._descend_until_contact(X, Y, YAW, pk.table_z - 0.015, tilt=tilt)
    p, _ = a.kin.fk(d.q_measured())
    print(f"contact: setpoint z {z * 1000:.1f} mm, FK tool point z {p[2] * 1000:.1f} mm at ({p[0]*1000:.1f}, {p[1]*1000:.1f})", flush=True)
    d.move_to(a._solve(X, Y, zh, YAW, True, tilt), speed=0.3)
    d.move_pose("scan", speed=0.5)
finally:
    park_and_shutdown(app)
