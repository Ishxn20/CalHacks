"""Local demo server. Camera frames are sent to Claude only after Start, and are never saved."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .robot import Camera, Robot
from .vision import ClaudeVision, GESTURES, VisionUnavailable, image_blocks

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.85
MAX_BODY = 4_600_000


class Controller:
    def __init__(self, vision, robot, available=True, clock=time.monotonic, confirmations=2, threshold=THRESHOLD):
        if confirmations not in (1, 2) or not 0.85 <= threshold <= 1:
            raise ValueError("Use 1 or 2 confirmations and a threshold from 0.85 to 1.")
        self.vision, self.robot, self.available, self.clock = vision, robot, available, clock
        self.confirmations, self.threshold = confirmations, threshold
        self.lock = threading.RLock()
        self.analysis = threading.Lock()
        self.enabled = self.auto = False
        self.generation = 0
        self.result = None
        self.candidate = None
        self.streak = self.neutral = 0
        self.latched = None
        self.last_pick = float("-inf")
        self.last_call = float("-inf")
        self.result_at = float("-inf")
        self.error = ""
        self.history = deque(maxlen=24)  # decision text only; never retain camera images

    def state(self):
        with self.lock:
            return {"enabled": self.enabled, "auto_pick": self.auto, "generation": self.generation,
                    "prediction": self.result, "streak": self.streak, "error": self.error,
                    "claude_available": self.available, "in_flight": self.analysis.locked(),
                    "mapping": GESTURES, "robot": self.robot.snapshot(), "history": list(self.history),
                    "confirmations": self.confirmations, "threshold": self.threshold}

    def configure(self, enabled, auto):
        if not isinstance(enabled, bool) or not isinstance(auto, bool):
            raise ValueError("enabled and auto_pick must be booleans.")
        with self.lock:
            if enabled and not self.available:
                raise ValueError("Set ANTHROPIC_API_KEY before starting recognition.")
            if enabled and self.robot.snapshot()["stopped"]:
                raise ValueError("Reset the arm before starting recognition.")
            self.enabled, self.auto = enabled, auto
            self.generation += 1
            self.candidate = self.result = self.latched = None
            self.streak = self.neutral = 0
            self.error = ""
            return self.state()

    def analyze(self, frames, generation):
        image_blocks(frames)             # validate before any API call
        if not self.analysis.acquire(blocking=False):
            return {"ignored": "Another frame batch is being analyzed."}
        try:
            with self.lock:
                if not self.enabled or generation != self.generation:
                    return {"ignored": "This camera session has stopped."}
                if self.clock() - self.last_call < 0.8:
                    return {"ignored": "Wait for the next camera sample."}
                self.last_call = self.clock()
            try:
                result = self.vision.classify(frames)
            except Exception:
                with self.lock:
                    if generation == self.generation:
                        self.candidate, self.streak, self.result = None, 0, None
                        self.error = "Claude could not read the cue. No pickup was requested."
                raise
            with self.lock:
                if not self.enabled or generation != self.generation:
                    return {"ignored": "The camera session changed while Claude was responding."}
                gesture = result["gesture"] if result.get("left_hand_on_brown_table") is True else "none"
                confident = (gesture in GESTURES and result.get("left_hand_on_brown_table") is True
                             and result["confidence"] >= self.threshold)
                self.streak = self.streak + 1 if confident and self.candidate == gesture else int(confident)
                self.candidate = gesture if confident else None
                self.neutral = self.neutral + 1 if gesture == "none" else 0
                if self.neutral >= 2:
                    self.latched = None
                self.error = ""
                self.result_at = self.clock()
                self.result = {**result, "gesture": gesture, **GESTURES.get(gesture, {}), "confirmed": self.streak >= self.confirmations,
                               "picked": False, "message": "Show the left-hand pose resting on the brown table."}
                if self.streak >= self.confirmations:
                    self.result["message"] = "Preset cue recognized."
                    if self.auto:
                        self._pick()
                self.history.append({"t": time.time(), **self.result})
                return self.state()
        finally:
            self.analysis.release()

    def _pick(self):
        if not self.enabled or not self.result or not self.result["confirmed"]:
            raise ValueError("A clear, recent left-hand pose on the brown table must be recognized before picking.")
        if self.clock() - self.result_at > 10:
            raise ValueError("That prediction is stale. Show the gesture again.")
        if self.latched == self.candidate:
            self.result["message"] = "Already requested. Lift the left hand briefly before repeating this cue."
        elif self.clock() - self.last_pick < 10:
            self.result["message"] = "Waiting for the pickup cooldown."
        elif self.robot.submit(self.result["instrument"]):
            self.last_pick, self.latched = self.clock(), self.candidate
            self.result.update(picked=True, message="Pickup requested.")
        else:
            self.result["message"] = "Arm is busy, stopped, holding a tool, or has no robot-camera picture."

    def prepare(self, generation):
        with self.lock:
            if generation != self.generation:
                raise ValueError("This camera session has changed.")
            self._pick()
            return self.state()

    def control(self, action):
        with self.lock:
            if action not in ("stop", "reset", "release"):
                raise ValueError("Unknown robot control.")
            if action in ("stop", "reset"):
                self.configure(False, False)  # invalidate any model response still in flight
            self.robot.control(action)
            return self.state()


def serve(controller, host="127.0.0.1", port=8790):
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The demo only listens on loopback.")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            host = self.headers.get("Host", "")
            name = host.split(":")[0]
            origin = self.headers.get("Origin")
            return name in {"localhost", "127.0.0.1"} and (origin is None or origin == "http://" + host)

        def do_GET(self):
            if not self.allowed():
                return self.send(403, {"error": "Request refused."})
            path = urlparse(self.path).path
            if path == "/":
                return self.send(200, (ROOT / "gesture" / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/camera.js":
                return self.send(200, (ROOT / "gesture" / "web" / "camera.js").read_bytes(), "text/javascript; charset=utf-8")
            if path == "/api/state":
                return self.send(200, controller.state())
            if path == "/api/robot.jpg":
                jpg = controller.robot.camera.jpeg()
                return self.send(200, jpg, "image/jpeg") if jpg else self.send(503, {"error": "Robot camera unavailable."})
            self.send(404, {"error": "Not found."})

        def do_POST(self):
            if not self.allowed():
                return self.send(403, {"error": "Request refused."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Invalid request size.")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object.")
                path = urlparse(self.path).path
                if path == "/api/configure":
                    result = controller.configure(body.get("enabled"), body.get("auto_pick", False))
                elif path == "/api/analyze":
                    result = controller.analyze(body.get("frames"), body.get("generation"))
                elif path == "/api/pick":
                    result = controller.prepare(body.get("generation"))
                elif path == "/api/control":
                    result = controller.control(body.get("action"))
                else:
                    return self.send(404, {"error": "Not found."})
                self.send(200, result)
            except VisionUnavailable as e:
                self.send(503, {"error": str(e)})
            except (ValueError, KeyError, TypeError) as e:
                self.send(400, {"error": str(e)})
            except Exception:
                self.send(500, {"error": "The demo could not complete that request. Check the server log."})

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def main():
    p = argparse.ArgumentParser(description="Left-hand pose on brown table -> Claude -> #10/#20/forceps")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--real-arm", action="store_true", help="connect and enable the calibrated physical arm")
    p.add_argument("--robot-camera", type=int, help="OpenCV index of the wrist camera (never the FaceTime camera)")
    p.add_argument("--env-file", type=Path, default=ROOT / ".env")
    p.add_argument("--model", default=None)
    p.add_argument("--confirm", type=int, choices=[1, 2], default=1)
    p.add_argument("--threshold", type=float, default=.90)
    args = p.parse_args()
    if not .85 <= args.threshold <= 1:
        p.error("--threshold must be between 0.85 and 1.")
    if args.env_file.exists():
        for line in args.env_file.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    sys.path[:0] = [str(ROOT / "arm"), str(ROOT / "perception")]
    camera = Camera(args.robot_camera)
    camera.start()
    try:
        robot = Robot(camera, real=args.real_arm)
    except Exception:
        camera.close()
        raise
    model = args.model or os.environ.get("GESTURE_CLAUDE_MODEL", "claude-sonnet-5-5")
    controller = Controller(ClaudeVision(model), robot, confirmations=args.confirm, threshold=args.threshold,
                            available=bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")))
    httpd = serve(controller, port=args.port)
    print(f"Gesture demo: http://127.0.0.1:{args.port} ({'REAL ARM' if args.real_arm else 'SIMULATED arm'})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        robot.close()
        camera.close()
