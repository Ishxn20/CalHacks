"""Tray-camera service the arm talks to (contract: arm/scrub_arm/vision.py).

    python -m scrub_vision.serve --camera 0                                   # classical blobs only, identity unchecked
    python -m scrub_vision.serve --camera 0 --weights runs/sandbox/weights/best.pt   # sandbox objects
    python -m scrub_vision.serve --camera 0 --weights runs/tools/weights/best.pt     # the 15 real tools
    python -m scrub_vision.serve --image data/scalpels.jpg                    # a still instead of a camera, for dry runs

GET /frame.jpg shows what the arm will see, with the grip lines drawn on.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from . import markers
from .targets import load_targets
from .grasp import Finder, combine, draw, pick_target

SAME_OBJECT_PX = 30      # detections this close across frames are the same object
STABLE_PX, STABLE_DEG = 3.0, 5.0


class StillCamera:
    def __init__(self, path: str) -> None:
        self.img = cv2.imread(path)
        if self.img is None:
            raise SystemExit(f"cannot read {path}")

    def frame(self):
        return self.img.copy()

    def close(self) -> None:
        pass


class Service:
    def __init__(self, camera, finder: Finder) -> None:
        self.camera, self.finder = camera, finder
        self._lock = threading.Lock()    # one YOLO call at a time

    def _find(self, frame):
        with self._lock:
            dets = self.finder.find(frame)
        return markers.without_markers(dets, frame) if self.finder.model is None else dets

    def detect(self, names: list[str] | None, frames: int = 5) -> dict:
        shots = []
        for _ in range(max(frames, 1)):
            shots.append(self.camera.frame())
            time.sleep(0.06)
        per = [self._find(f) for f in shots]
        h, w = shots[-1].shape[:2]
        out = {"found": False, "image": [w, h], "identity": "yolo" if self.finder.model else "unchecked",
               "markers": {str(i): m["center"] for i, m in markers.detect(shots[-1]).items()}}
        tgts = [t for t in (pick_target(d, names) for d in per) if t]
        if not tgts:
            return out
        same = [t for t in tgts if math.dist(t.center_px, tgts[0].center_px) < SAME_OBJECT_PX]
        det, spread_px, spread_deg = combine(same)
        out.update(
            found=True, **det.to_json(), spread_px=spread_px, spread_deg=spread_deg,
            stable=len(same) >= (len(per) + 1) // 2 and spread_px < STABLE_PX and spread_deg < STABLE_DEG,
            others=[{"name": d.name, "box_px": d.box_px} for d in per[-1]
                    if math.dist(d.center_px, det.center_px) >= SAME_OBJECT_PX],
        )
        return out

    def markers(self) -> dict:
        frame = self.camera.frame()
        found = {str(i): m for i, m in markers.detect(frame).items()}
        source = "aruco"
        if len(found) < 3 and len(load_targets()) >= 3:   # no printed markers in view: use the clicked targets
            found = {i: {"center": p, "side_px": None} for i, p in load_targets().items()}
            source = "clicked"
        return {"markers": found, "image": [frame.shape[1], frame.shape[0]], "source": source}

    def jpg(self) -> bytes:
        frame = self.camera.frame()
        dets = self._find(frame)
        return cv2.imencode(".jpg", draw(frame, dets, dets[0] if dets else None))[1].tobytes()


def make_handler(svc: Service):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            path = self.path.rstrip("/")
            if path == "/markers":
                self._json(200, svc.markers())
            elif path == "/frame.jpg":
                self._send(200, svc.jpg(), "image/jpeg")
            elif path in ("", "/health"):
                self._json(200, {"ok": True})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.rstrip("/") != "/detect":
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n)) if n else {}
                self._json(200, svc.detect(body.get("names"), int(body.get("frames", 5))))
            except (ValueError, TypeError) as e:
                self._json(400, {"error": f"bad request: {e}"})

    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int)
    src.add_argument("--image")
    ap.add_argument("--weights")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--port", type=int, default=8790)
    args = ap.parse_args()

    if args.camera is not None:
        from .camera import Camera
        cam = Camera(args.camera)
    else:
        cam = StillCamera(args.image)
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Service(cam, Finder(args.weights, args.conf))))
    httpd.daemon_threads = True
    print(f"tray camera service on http://127.0.0.1:{args.port} "
          f"({'YOLO ' + args.weights if args.weights else 'classical only, identity unchecked'}); Ctrl-C to quit", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()


if __name__ == "__main__":
    main()
