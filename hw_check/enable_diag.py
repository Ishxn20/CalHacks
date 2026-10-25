"""Enable all motors holding exactly where they are (no motion), then report each motor's run mode, feedback
status code, torque and fault report, then go limp. Arm must be resting folded (it holds still, then relaxes).
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib .venv/bin/python hw_check/enable_diag.py
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arm"))
from scrub_arm.app import build  # noqa: E402

RUN_MODE = 0x7005          # RobStride run_mode: 0 = MIT (operation control), 1 = PP, 2 = speed, 3 = current, 5 = CSP

app = build()
d = app.driver
bus = d.bus
d.start()
try:
    d.enable()
    time.sleep(1.0)                                   # the control loop streams hold-in-place setpoints meanwhile
    for name, m in zip(bus.names(), bus._motors):
        with d._bus_lock:
            st = m.get_state()
            try:
                mode = m.robstride_get_param_u8(RUN_MODE, 300)
            except Exception as e:  # noqa: BLE001
                mode = f"? ({e})"
            try:
                fault = m.robstride_get_fault_report()
            except Exception as e:  # noqa: BLE001
                fault = f"? ({e})"
        if st is None:
            print(f"{name:8s} NO FEEDBACK  run_mode {mode}  fault {fault}")
        else:
            print(f"{name:8s} pos {math.degrees(st.pos):+7.2f}  torq {st.torq:+6.2f} Nm  status 0x{st.status_code:x}  "
                  f"run_mode {mode}  t_mos {st.t_mos:.0f}C  fault {fault}")
finally:
    d.limp()
    d.shutdown()
