"""Hand-off calibration: find a hand-off pose where the wrist camera can see the held tool, then measure how much of
it the camera sees held vs taken, for tools/pick.py's "the tool was taken" check.

    DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/handoff_calib.py [TOOL]     (default: 20 blade)

1. picks the tool and presents it at the taught hand-off pose
2. tilts the wrist (joint5) in 10 deg steps either way and counts the tool's red pixels near the jaws at each
3. the tilt closest to the taught pose where the tool is clearly in view (or the best one) is offered as the new
   `handoff` pose (asks before saving poses.yaml; the old one is kept as `handoff_old`)
4. you take the tool out; it measures the count with the tool gone and writes the thresholds to
   tools/handoff_calib.json, which pick.py reads
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pick as P  # noqa: E402  (builds the arm app and opens the camera)

STEPS_DEG = [0, -10, 10, -20, 20, -30, 30, -40, 40]
CLEAR_PX = 8000          # this much red near the jaws: the tool is clearly in view
OUT = Path(__file__).with_name("handoff_calib.json")


def count(n=5) -> float:
    P.d.sleep(0.4)
    return float(np.median([P.held_red() for _ in range(n)]))


def main() -> None:
    r = P.app.catalog.resolve(sys.argv[1] if len(sys.argv) > 1 else "20 blade")
    d, cfg = P.d, P.app.cfg
    j5 = cfg.joint_index("joint5")
    d.start()
    d.enable()
    try:
        if not P.pick(r):
            return
        P.a._present(P.a.holding)
        q0 = d.q_command().copy()
        seen = []
        for step in STEPS_DEG:
            q = q0.copy()
            q[j5] += math.radians(step)
            if not (cfg.joints[j5].lower <= q[j5] <= cfg.joints[j5].upper):
                continue
            d.move_to(q, speed=0.3)
            n = count()
            seen.append((step, n, q))
            print(f"  wrist tilt {step:+4d} deg: {n:7.0f} red px near the jaws", flush=True)
        clear = [s for s in seen if s[1] >= CLEAR_PX]
        step, held, q = min(clear, key=lambda s: abs(s[0])) if clear else max(seen, key=lambda s: s[1])
        print(f"\n  best: tilt {step:+d} deg, {held:.0f} px with the tool held", flush=True)
        if held < 2000:
            print("  the camera can't see the held tool at any of these tilts: hand-off stays Enter-only")
            P.a._place_at(*P.a._picked_from)
            return
        d.move_to(q, speed=0.3)
        held = count(9)
        input("\n  Now take the tool out of the jaws (it can be pulled out), then press Enter ...")
        gone = count(9)
        print(f"  held {held:.0f} px, gone {gone:.0f} px", flush=True)
        if gone > 0.5 * held:
            print("  the count barely dropped when the tool was taken: not usable, nothing saved")
            return
        frac = round(float((held + gone) / 2 / held), 2)              # halfway between, as a fraction of held
        OUT.write_text(json.dumps({"min_px": round(0.5 * held), "gone_fraction": frac, "wrist_tilt_deg": step,
                                   "held_px": held, "gone_px": gone}, indent=1))
        print(f"  wrote {OUT.name}: taken when the count drops below {frac:.0%} of the held count")
        if step and input(f"  save this tilt ({step:+d} deg on joint5) as the hand-off pose? [y/N] ").strip().lower() == "y":
            poses = P.app.poses
            poses.set("handoff_old", poses.get("handoff").q)
            poses.set("handoff", q)
            poses.save()
            print("  saved (old pose kept as handoff_old)")
        P.a.holding = None
        d.gripper_open()
        d.move_pose("rest", speed=0.5)
    finally:
        P.camera.cap.release()
        P.park_and_shutdown(P.app)


if __name__ == "__main__":
    main()
