"""Private camera-pick adapter based on main at ccd58d88.

Keeps the existing tools/pick.py CLI untouched; uses the shared arm driver, detector and calibration.
"""
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

R = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/arm")
sys.path.insert(0, R + "/perception")
from scrub_arm.app import build                      # noqa: E402
from scrub_arm.catalog import Resolved  # noqa: E402
from scrub_arm.handoff import TaskFailed              # noqa: E402

TRIES = 3
STILL_THERE_MM = 20.0        # a complete detection of the tool this close to where it was: the pick missed
GONE_FRACTION = 0.3          # hand-off: the tool counts as taken when the red near the jaws drops below 30 % ...
GONE_HOLD_S = 0.5            # ... for this long (a hand passing in front doesn't count)
GONE_MIN_PX = 3000           # less red than this at the start: the camera can't see the held tool, Enter only
try:                         # measured by tools/handoff_calib.py
    _hc = __import__("json").loads((Path(R) / "tools" / "handoff_calib.json").read_text())
    GONE_FRACTION, GONE_MIN_PX = float(_hc["gone_fraction"]), float(_hc["min_px"])
except (OSError, ValueError, KeyError):
    pass

app = d = a = det = cap = None
_frame_source = None


def setup(application=None, camera_index=0, frame_source=None):
    """Import safely; explicitly initialize the arm and its single shared camera reader."""
    global app, d, a, det, cap, _frame_source
    from scrub_vision.red_parts import RedPartsDetector
    app = application if application is not None else build()
    d, a, det = app.driver, app.assistant, RedPartsDetector()
    _frame_source = frame_source
    if frame_source is None:
        cap = cv2.VideoCapture(camera_index)
        cap.set(3, 1920)
        cap.set(4, 1080)


def look():
    """Scan pose, jaws as they are; a fresh frame and the joints it was taken at."""
    d.move_pose("scan", speed=0.5)
    time.sleep(0.6)
    img = frame()
    if img is None:
        raise TaskFailed("the wrist camera gave no picture")
    return det.detect(img, d.q_measured())


def frame():
    """A fresh frame; the camera sometimes returns nothing, so retry, then reopen it."""
    global cap
    if _frame_source is not None:
        return _frame_source()
    for attempt in range(3):
        for _ in range(6):
            cap.read()
        for _ in range(5):
            ok, img = cap.read()
            if ok and img is not None:
                return img
            time.sleep(0.1)
        print("  (camera gave no picture; reopening it)", flush=True)
        cap.release()
        time.sleep(0.5)
        cap = cv2.VideoCapture(0)
        cap.set(3, 1920)
        cap.set(4, 1080)
    return None


def find(dets, fid):
    """The detection to pick for fid: named exactly first, then 'could be' (candidates); whole in view first."""
    named = [x for x in dets if x.fine_id == fid or fid in x.alt_ids]
    maybe = [x for x in dets if fid in x.candidates]
    pool = named or maybe
    return max(pool, key=lambda x: (x.exposure, x.confidence)) if pool else None


def pick(r: Resolved) -> bool:
    inst, fid = r.instrument, r.instrument.id
    for attempt in range(1, TRIES + 1):
        d.gripper_open()
        x = find(look(), fid)
        if x is None:
            print(f"  I don't see the {inst.name}")
            return False
        sure = "" if x.fine_id else f" (could be: {', '.join(x.candidates)})"
        print(f"  try {attempt}: {inst.name}{sure} at ({x.grasp_mm[0]:.0f}, {x.grasp_mm[1]:.0f}) mm, "
              f"{x.width_mm:.0f} mm wide there", flush=True)
        try:
            a._pick_at(x.grasp_mm[0], x.grasp_mm[1], x.yaw_deg, x.thickness_mm, fid=fid,
                       jaw_open_mm=inst.jaw_open_mm, width_mm=x.width_mm)
        except TaskFailed as e:
            print(f"  couldn't reach it: {e}")
            continue
        a.holding = r
        try:
            left = find(look(), fid)      # still lying where it was? (look() goes to the scan pose, still holding)
        except TaskFailed as e:
            print(f"  {e}; assuming it's held (the jaws stay closed)", flush=True)
            return True
        if left is not None and left.exposure == 1 and \
                np.hypot(left.grasp_mm[0] - x.grasp_mm[0], left.grasp_mm[1] - x.grasp_mm[1]) < STILL_THERE_MM:
            print("  missed: it's still on the table", flush=True)
            a.holding = None
            d.gripper_open()              # the camera sees it on the table, so the jaws are empty
            continue
        print(f"  holding the {inst.name}", flush=True)
        return True
    print(f"  gave up after {TRIES} tries")
    return False


def held_red() -> int:
    """Red pixels near the jaws in a fresh frame: the held tool (fingers masked out)."""
    from scrub_vision.red_parts import load_fingers, red_mask
    img = frame()
    if img is None:
        return -1
    m = red_mask(img).astype(bool) & ~load_fingers(img.shape)
    h, w = m.shape
    return int(m[h // 3:, w // 5: 4 * w // 5].sum())          # the lower middle of the view, around the jaws


def wait_until_taken(interactive: bool, timeout_s: float = 60.0, release_event=None) -> str:
    """Watch the wrist camera at the hand-off pose until the held tool is gone (the jaws can't stop it being pulled
    out, so the camera decides: user). Enter releases too. Returns 'taken', 'enter' or 'timeout'."""
    pressed = threading.Event()
    if interactive:
        threading.Thread(target=lambda: (input(), pressed.set()), daemon=True).start()
    base = np.median([held_red() for _ in range(5)])
    if base < GONE_MIN_PX:
        print(f"  (the camera can't see the tool in the jaws: {base:.0f} red px; press Enter to release)", flush=True)
    gone_since, t_end = None, time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        if pressed.is_set() or (release_event is not None and release_event.is_set()):
            return "enter"
        d.sleep(0.05)                     # honour Stop while waiting for the camera
        n = held_red()
        if base >= GONE_MIN_PX and 0 <= n < GONE_FRACTION * base:
            gone_since = gone_since or time.monotonic()
            if time.monotonic() - gone_since >= GONE_HOLD_S:
                return "taken"
        else:
            gone_since = None
    return "timeout"
