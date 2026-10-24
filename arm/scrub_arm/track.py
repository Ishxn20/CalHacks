"""Live wrist-camera window: see what the camera sees, teach the object's colour, find and grab it.

    python -m scrub_arm track [--camera N] [--sim]

Mouse: click the object to learn its colour.
Keys (window must be focused):
    h  hold (motors on where the arm is)      l  limp (SUPPORT THE ARM FIRST)
    d  point the gripper straight down        o  open jaws
    t t  record table height (fingertips touching the table; press twice to confirm)
    c  calibrate: object directly under the jaws, gripper pointing down ~5-8 cm above it
    g  automatic: look around with the gripper angled at the table, aim, approach, point down, grab
    p  park: go back to the folded rest pose
    s  stop (freeze)     r  reset after a stop/fault     q  quit (parks at rest first)
"""
from __future__ import annotations

import logging
import math
import threading
import time

import numpy as np

from .app import App
from .config import save_table_z
from .grasp import point_down, tilt_from_down
from .teach import park_and_shutdown
from .vision import Camera, SimCamera, detect, learn_color, load_vision, match_fraction
from .visual_grasp import _check_clear_of_table, calibrate_camera, look_and_grab

log = logging.getLogger(__name__)
WINDOW = "scrub_arm track"


class _Jobs:
    """Run one arm action at a time off the UI thread and keep its outcome for the overlay."""

    def __init__(self) -> None:
        self.busy: str | None = None
        self._status = ""
        self.status = "ready - click the window, then press h to turn the motors on"

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, text: str) -> None:
        if text != self._status:
            print(f"  {text}", flush=True)  # mirror to the terminal: easier to read and copy
        self._status = text

    def run(self, name: str, fn) -> None:
        if self.busy:
            self.status = f"busy with {self.busy}"
            return
        self.busy, self.status = name, f"{name}..."

        def work():
            try:
                out = fn()
                self.status = f"{name}: {out}" if out else f"{name}: done"
            except Exception as e:  # noqa: BLE001 - show any failure in the window
                self.status = f"{name} FAILED: {e}"
                log.warning("%s failed: %s", name, e)
            finally:
                self.busy = None

        threading.Thread(target=work, daemon=True).start()


def _draw(frame, det, mask, vcfg, lines):
    import cv2

    img = frame.copy()
    if vcfg.target_px is not None:
        u, v = (int(x) for x in vcfg.target_px)
        cv2.drawMarker(img, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)
    if det is not None:
        cv2.drawContours(img, [det.hull], -1, (0, 255, 0), 2)
        cv2.circle(img, (int(det.center[0]), int(det.center[1])), 5, (0, 0, 255), -1)
    if mask is not None:
        h, w = img.shape[:2]
        thumb = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (w // 4, h // 4))
        img[h - h // 4:, w - w // 4:] = thumb
    for i, text in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def run_track(app: App, camera_index: int | None = None, sim: bool = False) -> None:
    import cv2

    d, kin = app.driver, app.kin
    vcfg = load_vision(app.cfg.path.parent / "vision.yaml" if app.cfg.path else None)
    if camera_index is not None and camera_index != vcfg.camera_index:
        vcfg.camera_index = camera_index
        vcfg.save()

    d.start()
    if sim:
        d.enable()
        d.move_to(np.radians([0, 68, 46, -48, 0, 0]))
        point_down(d, kin)
        tip = kin.fk(d.q_command())[:3, 3]
        if app.cfg.grab.table_z is None:
            app.cfg.grab.table_z = float(tip[2]) - 0.07
        cam = SimCamera(d, kin, app.cfg.grab.table_z, obj_xy=tip[:2] + [0.02, -0.015])
    else:
        cam = Camera(vcfg.camera_index)
    cam.start()

    jobs = _Jobs()
    last = {"frame": None}
    pending: dict[str, float] = {}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and last["frame"] is not None:
            vcfg.hsv_lo, vcfg.hsv_hi, note = learn_color(last["frame"], x, y)
            vcfg.save()
            share = match_fraction(last["frame"], vcfg)
            warn = (f" - WARNING: matches {share:.0%} of the image, it will confuse the object with the "
                    "background; click the middle of the object or use a contrasting surface") if share > 0.10 else ""
            jobs.status = f"learned: {note}{warn}"

    def record_table():
        q = d.q_measured() if d.enabled else d.read_fresh()[:-1]
        z = float(kin.fk(q)[2, 3])
        save_table_z(app.cfg, z)
        return f"table height {z * 1000:.1f} mm"

    def hold():
        d.enable()
        return "motors on"

    def down():
        d.clear_abort()
        return f"pointing down ({math.degrees(point_down(d, kin)):.1f} deg off)"

    def park():
        d.clear_abort()
        if not d.enabled:
            d.enable()
        q_rest = d.pose_q("rest")
        if app.cfg.grab.table_z is not None:
            _check_clear_of_table(kin, d.q_command(), q_rest, app.cfg.grab.table_z)
        d.move_to(q_rest, speed=0.5)
        return "at rest"

    def opened():
        d.clear_abort()
        d.gripper_open()
        return "open"

    def grab():
        d.clear_abort()
        def say(msg):
            jobs.status = f"find & grab: {msg}"
        return str(look_and_grab(d, kin, cam, vcfg, status=say))

    def calibrate():
        d.clear_abort()
        return calibrate_camera(d, kin, cam, vcfg)

    actions = {
        ord("h"): ("hold", hold),
        ord("d"): ("point down", down),
        ord("o"): ("open", opened),
        ord("p"): ("park", park),
        ord("c"): ("calibrate", calibrate),
        ord("g"): ("find & grab", grab),
    }

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW, on_mouse)
    try:
        while True:
            frame = cam.latest()
            if frame is None:
                time.sleep(0.01)
                continue
            last["frame"] = frame
            det, mask = detect(frame, vcfg)

            q = d.q_measured()
            if d.enabled and np.isnan(q).any():
                arm = "motors ON (no position feedback yet)"
            elif d.enabled:
                z = kin.fk(q)[2, 3]
                above = f"{(z - app.cfg.grab.table_z) * 1000:.0f} mm above table" if app.cfg.grab.table_z is not None \
                    else f"z={z * 1000:.0f} mm (table not set: t)"
                arm = f"motors ON | {above} | tilt {math.degrees(tilt_from_down(kin, q)):.0f} deg"
            else:
                arm = "motors OFF (h = hold)"
            seen = (f"object at ({det.center[0]:.0f}, {det.center[1]:.0f})" if det is not None
                    else "object not seen" if vcfg.hsv_lo else "click the object to learn its colour")
            lines = [arm, seen + ("" if vcfg.calibrated else " | not calibrated (c)"),
                     jobs.status[:110], "h hold  l limp  d down  o open  p park  t table  c calibrate  g grab  s stop  q quit"]
            if d.fault:
                lines.insert(0, f"FAULT: {d.fault} (r to reset)")
            cv2.imshow(WINDOW, _draw(frame, det, mask, vcfg, lines))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                d.stop()
                jobs.status = "stopped"
            elif key == ord("r"):
                d.clear_fault()
                jobs.status = "reset"
            elif key == ord("t"):
                # Two presses: a stray `t` once moved the table height 5 cm and ruined every grab.
                if time.monotonic() - pending.get("t", 0.0) < 3.0:
                    pending.pop("t")
                    jobs.run("table", record_table)
                else:
                    pending["t"] = time.monotonic()
                    jobs.status = "fingertips touching the table? press t again within 3 s to record its height"
            elif key == ord("l"):
                if jobs.busy:
                    jobs.status = "stop first (s)"
                else:
                    d.limp()
                    jobs.status = "limp - move it by hand, then h"
            elif key in actions:
                name, fn = actions[key]
                jobs.run(name, fn)
    finally:
        if jobs.busy:
            d.stop()
            time.sleep(0.3)
        cv2.destroyAllWindows()
        cam.stop()
        park_and_shutdown(app)
