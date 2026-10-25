"""Clear latched RobStride faults (e.g. joint2's stall-overload lockout) and show each motor's fault report
before and after. Motors stay disabled throughout: nothing moves.
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib .venv/bin/python hw_check/clear_faults.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arm"))
from scrub_arm.app import build  # noqa: E402

bus = build().driver.bus
bus.connect()
for name, m in zip(bus.names(), bus._motors):
    before = m.robstride_get_fault_report()
    if before != (0, 0):
        m.clear_error()
        time.sleep(0.1)
        print(f"{name:8s} fault {before} -> cleared -> {m.robstride_get_fault_report()}")
    else:
        print(f"{name:8s} no fault")
bus.disable()
bus.close()
