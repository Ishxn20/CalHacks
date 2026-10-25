"""The wrist-camera pick and hand-off proven on the real arm (tools/pick.py), shared by that CLI and the server's jobs.

Each try: scan pose -> find the tool (red_parts) -> pick (its jaw width, the real pick motion) -> back to the scan
pose and look: if the tool is still lying where it was, that was a miss, so open and try again from a fresh scan.
Hand-off: the tool counts as taken when the wrist camera stops seeing it in the jaws.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from .catalog import Resolved
from .handoff import Phase, TaskFailed

TRIES = 3
STILL_THERE_MM = 20.0        # a complete detection of the tool this close to where it was: the pick missed
GONE_FRACTION = 0.3          # hand-off: the tool counts as taken when the red near the jaws drops below 30 % ...
GONE_HOLD_S = 0.5            # ... for this long (a hand passing in front doesn't count)
GONE_MIN_PX = 3000           # less red than this at the start: the camera can't see the held tool, Enter only
try:                         # measured by tools/handoff_calib.py
    _hc = json.loads((Path(__file__).resolve().parents[2] / "tools" / "handoff_calib.json").read_text())
    GONE_FRACTION, GONE_MIN_PX = float(_hc["gone_fraction"]), float(_hc["min_px"])
except (OSError, ValueError, KeyError):
    pass


ROOT = Path(__file__).resolve().parents[2]


def _open_camera(index: int):
    cap = cv2.VideoCapture(index)
    cap.set(3, 1920)
    cap.set(4, 1080)
    return cap


class CameraPickup:
    def __init__(self, assistant, camera_index: int = 0, detector=None):
        if detector is None:
            from scrub_vision.red_parts import RedPartsDetector
            detector = RedPartsDetector()
        self.a, self.d, self.det = assistant, assistant.driver, detector
        self.camera_index = camera_index
        cam = yaml.safe_load((ROOT / "config" / "camera.yaml").read_text())
        self.size = (int(cam["width"]), int(cam["height"]))      # the calibration is only valid at this size
        self.cap = _open_camera(camera_index)

    def close(self) -> None:
        self.cap.release()

    def _say(self, msg: str) -> None:
        print("  " + msg, flush=True)
        self.a._log(msg)

    def frame(self):
        """A fresh frame at the calibrated size; the camera sometimes returns nothing, or a smaller picture when another
        app (a browser preview) opens it, so retry, then reopen it (which asks for the calibrated size again)."""
        for _ in range(3):
            for _ in range(6):
                self.cap.read()
            for _ in range(5):
                ok, img = self.cap.read()
                if ok and img is not None:
                    if (img.shape[1], img.shape[0]) == self.size:
                        return img
                    self._say(f"camera sent {img.shape[1]}x{img.shape[0]}, calibrated for {self.size[0]}x"
                              f"{self.size[1]}: is another app using the wrist camera?")
                    break
                self.d.sleep(0.1)
            self._say("reopening the wrist camera")
            self.cap.release()
            self.d.sleep(0.5)
            self.cap = _open_camera(self.camera_index)
        return None

    def look(self):
        """Scan pose, jaws as they are; the detections in a fresh frame and the joints it was taken at."""
        self.d.move_pose("scan", speed=0.5)
        self.d.sleep(0.6)
        img = self.frame()
        if img is None:
            raise TaskFailed("the wrist camera gave no usable picture; close anything else using it")
        return self.det.detect(img, self.d.q_measured())

    @staticmethod
    def find(dets, fid):
        """The detection to pick for fid: named exactly first, then 'could be' (candidates); whole in view first."""
        named = [x for x in dets if x.fine_id == fid or fid in x.alt_ids]
        maybe = [x for x in dets if fid in x.candidates]
        pool = named or maybe
        return max(pool, key=lambda x: (x.exposure, x.confidence)) if pool else None

    def pick(self, r: Resolved) -> bool:
        a, d = self.a, self.d
        inst, fid = r.instrument, r.instrument.id
        for attempt in range(1, TRIES + 1):
            a.phase = Phase.PICKING
            d.gripper_open()
            dets = self.look()
            x = self.find(dets, fid)
            if x is None:
                seen = "; ".join(f"{d.fine_id or ('could be ' + '/'.join(d.candidates) if d.candidates else 'not ours')}"
                                 f" {d.length_mm:.0f} mm{'' if d.exposure == 1 else ' (cut off)'}" for d in dets)
                self._say(f"I don't see the {inst.name}; the camera sees: {seen or 'no red parts'}")
                return False
            sure = "" if x.fine_id else f" (could be: {', '.join(x.candidates)})"
            self._say(f"try {attempt}: {inst.name}{sure} at ({x.grasp_mm[0]:.0f}, {x.grasp_mm[1]:.0f}) mm, "
                      f"{x.width_mm:.0f} mm wide there")
            try:
                a._pick_at(x.grasp_mm[0], x.grasp_mm[1], x.yaw_deg, x.thickness_mm, fid=fid,
                           jaw_open_mm=inst.jaw_open_mm, width_mm=x.width_mm)
            except TaskFailed as e:
                self._say(f"couldn't reach it: {e}")
                continue
            a.holding = r
            a.phase = Phase.VERIFYING
            try:
                left = self.find(self.look(), fid)  # still lying where it was? (look() goes to the scan pose, still holding)
            except TaskFailed as e:
                self._say(f"{e}; assuming it's held (the jaws stay closed)")
                return True
            if left is not None and left.exposure == 1 and \
                    np.hypot(left.grasp_mm[0] - x.grasp_mm[0], left.grasp_mm[1] - x.grasp_mm[1]) < STILL_THERE_MM:
                self._say("missed: it's still on the table")
                a.holding = None
                d.gripper_open()              # the camera sees it on the table, so the jaws are empty
                continue
            self._say(f"holding the {inst.name}")
            return True
        self._say(f"gave up after {TRIES} tries")
        return False

    def held_red(self) -> int:
        """Red pixels near the jaws in a fresh frame: the held tool (fingers masked out)."""
        from scrub_vision.red_parts import load_fingers, red_mask
        img = self.frame()
        if img is None:
            return -1
        m = red_mask(img).astype(bool) & ~load_fingers(img.shape)
        h, w = m.shape
        return int(m[h // 3:, w // 5: 4 * w // 5].sum())          # the lower middle of the view, around the jaws

    def wait_until_taken(self, interactive: bool = False, timeout_s: float = 60.0) -> str:
        """Watch the wrist camera at the hand-off pose until the held tool is gone (the jaws can't stop it being pulled
        out, so the camera decides: user). Enter (CLI) or a release command (voice/HTTP) releases too.
        Returns 'taken', 'enter', 'release command' or 'timeout'."""
        pressed = threading.Event()
        if interactive:
            threading.Thread(target=lambda: (input(), pressed.set()), daemon=True).start()
        base = np.median([self.held_red() for _ in range(5)])
        if base < GONE_MIN_PX:
            self._say(f"the camera can't see the tool in the jaws ({base:.0f} red px); "
                      + ("press Enter to release" if interactive else "say \"got it\" to release"))
        gone_since, t_end = None, time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            self.d.sleep(0.01)            # raises if the operator stops the arm
            if pressed.is_set():
                return "enter"
            if self.a._release.is_set():
                return "release command"
            n = self.held_red()
            if base >= GONE_MIN_PX and 0 <= n < GONE_FRACTION * base:
                gone_since = gone_since or time.monotonic()
                if time.monotonic() - gone_since >= GONE_HOLD_S:
                    return "taken"
            else:
                gone_since = None
        return "timeout"
