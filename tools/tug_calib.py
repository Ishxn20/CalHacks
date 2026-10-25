"""Measure how much the arm's joints deflect at the hand-off pose on their own vs when someone pulls the tool.

    DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/python tools/tug_calib.py [TOOL]      (default: 20 blade)

Picks the tool (tools/pick.py), presents it at the hand-off pose, then records each joint's deviation from its
setpoint (minus the resting value: gravity sag) for 10 s untouched and 10 s while you pull it like a surgeon taking
it (several firm tugs). Prints the largest deviation per joint for both, and a threshold halfway between, for
arm.yaml handoff.tug. Then puts the tool back.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pick as P  # noqa: E402  (builds the arm app and opens the camera)

SECONDS = 10.0


def torque() -> np.ndarray:
    with P.d._lock:
        return P.d._tau.copy()                          # all 7 motors, N m (last = gripper)


def record(label: str, base: np.ndarray, tbase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print(f"\n  {label}: recording {SECONDS:.0f} s ...", flush=True)
    peak, tpeak = np.zeros(len(base)), np.zeros(len(tbase))
    t_end = time.monotonic() + SECONDS
    while time.monotonic() < t_end:
        P.d.sleep(0.01)
        peak = np.fmax(peak, np.abs(P.d.tracking_deviation() - base))
        tpeak = np.fmax(tpeak, np.abs(torque() - tbase))
    return np.degrees(peak), tpeak


def main() -> None:
    text = sys.argv[1] if len(sys.argv) > 1 else "20 blade"
    r = P.app.catalog.resolve(text)
    P.d.start()
    P.d.enable()
    try:
        if not P.pick(r):
            return
        P.a._present(P.a.holding)
        P.d.sleep(1.0)                                       # let the move settle
        samples, tsamples = [], []
        for _ in range(50):
            P.d.sleep(0.02)
            samples.append(P.d.tracking_deviation())
            tsamples.append(torque())
        base, tbase = np.nanmean(samples, axis=0), np.nanmean(tsamples, axis=0)
        input("\n  Holding the tool at the hand-off pose. Don't touch the arm. Press Enter to start ...")
        still, tstill = record("hands off", base, tbase)
        input("\n  Now pull the tool like you're taking it (several firm tugs, don't yank it out). Press Enter to start ...")
        pulled, tpulled = record("pulling", base, tbase)
        names = [j.name for j in P.app.cfg.joints]
        print("\n  joint     still (deg)   pulled (deg)   still torque (Nm)   pulled torque (Nm)")
        for n, s, p, ts, tp in zip(names + ["gripper"], list(still) + [0], list(pulled) + [0], tstill, tpulled):
            print(f"  {n:8s}  {s:9.2f}     {p:9.2f}       {ts:9.3f}          {tp:9.3f}")
        tug = [i for i, n in enumerate(names) if n in P.app.cfg.handoff.tug.joints]
        s, p = still[tug].max(), pulled[tug].max()
        print(f"\n  tug joints {[names[i] for i in tug]}: still max {s:.2f} deg, pulled max {p:.2f} deg")
        print(f"  suggested deviation_deg: {(s + p) / 2:.1f}" if p > 2 * s else
              "  pull didn't clearly beat the resting noise: pull harder, or use other joints")
        P.a._place_at(*P.a._picked_from)
    finally:
        P.camera.cap.release()
        P.park_and_shutdown(P.app)


if __name__ == "__main__":
    main()
