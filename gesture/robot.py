"""One worker owns the arm; gesture.picker adapts main's calibrated camera-checked pick."""
from __future__ import annotations

import threading
import time

from .vision import GESTURES

ALLOWED = {v["instrument"] for v in GESTURES.values()}


class Camera:
    """One OpenCV reader shared by the robot preview, detector and hand-off watcher."""
    def __init__(self, index):
        self.index = index
        self._lock = threading.Lock()
        self._frame = None
        self._at = 0
        self.error = "Robot camera is not connected."
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.index is not None:
            self.thread = threading.Thread(target=self._read, daemon=True, name="robot-camera")
            self.thread.start()

    def _read(self):
        import cv2
        cap = cv2.VideoCapture(self.index)
        cap.set(3, 1920)
        cap.set(4, 1080)
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if ok:
                    with self._lock:
                        self._frame, self._at = frame, time.monotonic()
                    self.error = ""
                else:
                    self.error = "Robot camera has no fresh frames. Check its index and camera permission."
                    self.stop_event.wait(0.2)
        finally:
            cap.release()

    def frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None and time.monotonic() - self._at < 2 else None

    def jpeg(self):
        import cv2
        frame = self.frame()
        if frame is None:
            return None
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)


class Robot:
    def __init__(self, camera: Camera, real=False):
        self.camera, self.real = camera, real
        self._lock = threading.RLock()
        self._busy = False
        self._stopped = False
        self._release = threading.Event()
        self._state = {"phase": "idle", "tool": None, "message": "Ready.", "error": None}
        self._thread = None
        self.picker = None
        if real:
            if camera.index is None:
                raise ValueError("--real-arm requires --robot-camera INDEX.")
            from scrub_arm.app import build, check_poses
            from gesture import picker as pick
            app = build()
            problems = check_poses(app)
            if problems:
                raise ValueError("Arm setup incomplete: " + "; ".join(problems))
            pick.setup(application=app, frame_source=camera.frame)
            self.picker = pick
            app.driver.start()
            app.driver.enable()

    def snapshot(self):
        with self._lock:
            return {**self._state, "busy": self._busy, "stopped": self._stopped, "simulated": not self.real,
                    "camera_error": self.camera.error}

    def _set(self, **values):
        with self._lock:
            self._state.update(values)

    def submit(self, instrument):
        if instrument not in ALLOWED:
            raise ValueError("This demo only picks #10, #20 and fine vascular forceps.")
        with self._lock:
            if self._busy or self._stopped or self._state["tool"]:
                return False
            if self.real and (self.picker.d.fault or self.camera.frame() is None):
                return False
            if self.real:
                self.picker.d.clear_abort()
            self._busy = True
            self._release.clear()
            self._set(phase="picking", tool=instrument, error=None, message="Picking the selected tool.")
            self._thread = threading.Thread(target=self._run, args=(instrument,), daemon=True, name="gesture-pick")
            self._thread.start()
            return True

    def _run(self, instrument):
        try:
            if self.real:
                p = self.picker
                resolved = p.app.catalog.resolve(instrument)
                if not p.pick(resolved):
                    self._set(phase="idle", tool=None, message="Tool not picked. Check the table and try again.")
                    return
                p.a._present(resolved)
            else:
                if self._release.wait(0.5) and self._stopped:
                    return
            with self._lock:
                if self._stopped:
                    return
                self._set(phase="presenting", message="Take the tool. The robot camera watches; Release also opens the jaws.")
            if self.real:
                how = self.picker.wait_until_taken(False, timeout_s=60, release_event=self._release)
                with self._lock:
                    if self._stopped:
                        return
                    if how != "timeout":
                        # Stop also wakes the wait event; it must never be mistaken for permission to open.
                        self.picker.a._let_go(wait_s=0)
                if how == "timeout":
                    self.picker.a._place_at(*self.picker.a._picked_from)
                    self._set(message="Not taken in 60 seconds; returned to the table.")
                else:
                    self.picker.d.sleep(0.4)
                    self.picker.d.move_pose("rest", speed=0.5)
                    self._set(message="Tool handed over.")
            else:
                taken = self._release.wait(60)
                self._set(message="Simulated handoff complete." if taken else "Simulated timeout: tool returned.")
            if not self._stopped:
                self._set(phase="idle", tool=None)
        except Exception as e:
            self._set(phase="fault", error=str(e), message="Arm task interrupted. Check the arm before resetting.")
        finally:
            with self._lock:
                self._busy = False
                if self._stopped:
                    self._set(phase="stopped", message="Stopped. Reset before starting again.")

    def control(self, action):
        with self._lock:
            if action == "stop":
                self._stopped = True
                if self.real:
                    self.picker.a.stop()
                self._release.set()
                self._set(phase="stopped", message="Stopped. Reset before starting again.")
            elif action == "release":
                if self._state["phase"] != "presenting":
                    raise ValueError("Release is available only while presenting a tool.")
                self._release.set()
            elif action == "reset":
                if self._busy:
                    raise ValueError("Wait for the current task to stop before resetting.")
                if self.real and self.picker.a.holding is not None:
                    raise ValueError("The arm is still holding a tool. Recover it with the teach tool before resetting.")
                if self.real:
                    self.picker.a.reset()
                self._stopped = False
                self._set(phase="idle", tool=None, error=None, message="Ready.")
            else:
                raise ValueError("Unknown robot control.")
        return self.snapshot()

    def close(self):
        self.control("stop")
        if self._thread:
            self._thread.join(timeout=5)
        if self.real:
            from scrub_arm.teach import park_and_shutdown
            park_and_shutdown(self.picker.app)
