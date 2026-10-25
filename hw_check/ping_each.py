"""Ping each motor and read its position (motors stay off: nothing moves)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "arm"))
from scrub_arm.app import build  # noqa: E402

bus = build().driver.bus
bus.connect()
for name, err in bus.ping().items():
    print(f"{name:8s} {'answers' if not err else 'NO ANSWER: ' + str(err)}", flush=True)
