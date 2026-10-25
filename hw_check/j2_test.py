"""Slow joint-2-only test: from rest, raise joint2 by +15 deg at ~5 deg/s, logging setpoint, measured angle and
motor torque every 0.1 s, so we can tell "blocked / overloaded" (torque near the 36 Nm limit, angle stuck) from
"not commanded" (torque ~0). Stops early if joint2 lags > 8 deg. Then returns to the start and goes limp
(the start is the folded rest pose, so nothing drops).
Run from the repo root with nothing else on the CAN adapter:
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib .venv/bin/python hw_check/j2_test.py
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arm"))
from scrub_arm.app import build  # noqa: E402

RISE, RATE, LAG = 15.0, 5.0, 8.0          # deg, deg/s, deg

app = build()
d = app.driver
d.start()
try:
    d.enable()
    time.sleep(0.6)
    q0 = d.q_command()
    print(f"start joints (deg): {np.round(np.degrees(q0), 1)}")
    print("   t   j2 cmd  j2 meas    lag   j2 torque Nm   j3 meas")
    t0, worst = time.time(), 0.0
    while True:
        t = time.time() - t0
        rise = min(RISE, RATE * t)
        q = q0.copy()
        q[1] += math.radians(rise)
        with d._lock:                     # stream the setpoint directly: a slow ramp, no trajectory
            d._q_cmd, d._qd_cmd = q, np.zeros(len(q))
        time.sleep(0.1)
        with d._lock:
            pos, tau = d._pos.copy(), d._tau.copy() if getattr(d, "_tau", None) is not None else np.full(7, np.nan)
        lag = math.degrees(q[1] - pos[1])
        worst = max(worst, abs(lag))
        print(f"{t:5.1f} {math.degrees(q[1]):8.2f} {math.degrees(pos[1]):8.2f} {lag:6.2f} {tau[1]:14.2f} "
              f"{math.degrees(pos[2]):9.2f}")
        if abs(lag) > LAG:
            print(f"STOP: joint2 lags {lag:.1f} deg")
            break
        if rise >= RISE and t > RISE / RATE + 1.0:
            break
    print(f"worst lag {worst:.1f} deg; returning to start")
    with d._lock:
        d._q_cmd = d._pos[:-1].copy()     # hold where it actually is first
    time.sleep(0.3)
    d.clear_fault()
    d.move_to(q0, speed=0.2)
finally:
    time.sleep(0.5)
    d.limp()
    d.shutdown()
