"""Open the gripper in small steps from wherever it is, stopping at the first sign of resistance.

The saved open/closed angles don't match the motor's current reading, so this never commands them: it steps
+STEP deg at a time (open = increasing angle, as in arm.yaml) and stops when the motor lags its command by
more than STALL deg (hard stop or contact), then backs off. Joints 1-6 just hold where they are; at the end
everything goes limp again (the arm is resting at its zero pose, so nothing falls).
Run from the repo root: DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib .venv/bin/python hw_check/open_gripper.py [max_deg]
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arm"))
from scrub_arm.app import build  # noqa: E402

STEP, STALL, MAX = 8.0, 10.0, float(sys.argv[1]) if len(sys.argv) > 1 else 150.0

app = build()
d = app.driver
d.start()
try:
    d.enable()
    time.sleep(0.5)
    start = math.degrees(d.gripper_measured())
    cmd = start
    print(f"gripper starts at {start:.1f} deg; opening up to +{MAX:.0f} deg")
    while cmd - start < MAX:
        cmd += STEP
        d.set_gripper(math.radians(cmd))
        time.sleep(0.4)
        meas = math.degrees(d.gripper_measured())
        print(f"  cmd {cmd:7.1f}  measured {meas:7.1f}")
        if cmd - meas > STALL:
            cmd = meas - 5.0
            d.set_gripper(math.radians(cmd))
            time.sleep(0.4)
            print(f"  resistance -> backed off to {cmd:.1f} deg")
            break
    print(f"final gripper angle {math.degrees(d.gripper_measured()):.1f} deg (opened {cmd - start:+.1f})")
finally:
    d.limp()
    d.shutdown()
