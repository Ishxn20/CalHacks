"""Read-only: each motor's fault report and state (position, torque, anything else get_state exposes).
Never enables torque. Run from the repo root with nothing else using the CAN adapter:
    DYLD_FALLBACK_LIBRARY_PATH=/usr/local/lib .venv/bin/python hw_check/motor_diag.py
"""
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arm"))
from scrub_arm.config import load_arm_config  # noqa: E402
from motorbridge import Controller  # noqa: E402

cfg = load_arm_config()
ctrl = Controller(cfg.channel)
specs = [(j.name, j.motor_id, j.feedback_id, j.model) for j in cfg.joints]
specs.append(("gripper", cfg.gripper.motor_id, cfg.gripper.feedback_id, cfg.gripper.model))
motors = [(name, ctrl.add_robstride_motor(mid, fid, model)) for name, mid, fid, model in specs]
for name, m in motors:
    try:
        m.robstride_ping()
    except Exception as e:  # noqa: BLE001
        print(f"{name}: ping failed: {e}")
time.sleep(0.3)
for name, m in motors:
    try:
        fault = m.robstride_get_fault_report()
    except Exception as e:  # noqa: BLE001
        fault = f"(could not read: {e})"
    limits = {}
    for pname, pid in (("limit_torque", 0x700B), ("limit_cur", 0x7018), ("limit_spd", 0x7017)):
        try:
            limits[pname] = round(m.robstride_get_param_f32(pid), 3)
        except Exception as e:  # noqa: BLE001
            limits[pname] = f"? ({e})"
    print(f"{name:8s} limits {limits}")
    st = m.get_state()
    fields = {k: getattr(st, k) for k in dir(st) if not k.startswith("_") and not callable(getattr(st, k))}
    pos = fields.get("pos")
    print(f"{name:8s} pos {math.degrees(pos) if pos is not None else float('nan'):+8.2f} deg  "
          f"state {fields}  fault {fault}")
ctrl.close() if hasattr(ctrl, "close") else None
