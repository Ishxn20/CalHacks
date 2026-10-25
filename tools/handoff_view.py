"""Pick a tool, present it, and print how much of it the wrist camera sees near the jaws every second for 30 s
(take it out partway through). Saves the first frame to hw_check/real/handoff_view.jpg. Then rest.
    DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/handoff_view.py [TOOL]
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pick as P  # noqa: E402

r = P.app.catalog.resolve(sys.argv[1] if len(sys.argv) > 1 else "20 blade")
P.d.start()
P.d.enable()
try:
    if P.pick(r):
        P.a._present(P.a.holding)
        P.d.sleep(0.5)
        for _ in range(4):
            P.cap.read()
        ok, img = P.cap.read()
        out = Path(P.R) / "hw_check/real/handoff_view.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img)
        print("  presenting: take the tool out whenever you like", flush=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 30:
            print(f"  t={time.monotonic() - t0:4.1f}s  red near jaws: {P.held_red()}", flush=True)
            time.sleep(0.7)
        P.a._let_go()
        P.d.move_pose("rest", speed=0.5)
finally:
    P.cap.release()
    P.park_and_shutdown(P.app)
