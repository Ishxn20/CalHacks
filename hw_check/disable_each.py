"""Which motor doesn't acknowledge 'disable'? Disable each one alone (motors are already off: nothing moves)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arm"))
from scrub_arm.app import build  # noqa: E402

bus = build().driver.bus
bus.connect()
for name, m in zip(bus.names(), bus._motors):
    for i in range(3):
        try:
            m.disable()
            print(f"{name:8s} disable ok", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"{name:8s} disable FAILED ({e})", flush=True)
            time.sleep(0.2)
try:
    bus._ctrl.close()
except Exception as e:  # noqa: BLE001
    print("close:", e)
