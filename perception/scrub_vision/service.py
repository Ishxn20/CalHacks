"""Perception service for the arm.

    python -m scrub_vision.service [--weights weights/pile_seg.pt] [--frame some.jpg] [--port 8770]

    POST /plan {"instrument": id}            -> declutter plan (see planner.py)
    POST /verify {"expected": id, "slot": _} -> {"ok", "detected", "confidence"} (arm's verify hook)
    POST /count                              -> {"instruments": [{"id", "confident"}], "unrecognised": n}: what the camera sees
    POST /reset_staging                      -> forget which staging spots are used
    GET  /debug                              -> annotated JPEG of the last analysed frame
    GET  /status
Reads repo-root config/camera.yaml (calibration) and arm/config/arm.yaml (pick.staging_mm).
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

from .identify import Detection, annotate, identify, segment_background
from .planner import plan
from .templates import build_templates, load_instruments

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_MM_PER_PX = 0.5


class BackgroundSegmenter:
    """Fallback: anything that isn't table surface, any colour; no class (identify() decides).
    Touching instruments merge, so it only works on spread-out instruments."""

    def __call__(self, bgr: np.ndarray, gripper_mask: np.ndarray | None) -> list[tuple[np.ndarray, str | None, float]]:
        return [(m, None, 1.0) for m in segment_background(bgr, gripper_mask=gripper_mask)]


class YoloSegmenter:
    def __init__(self, weights: str | Path, conf: float = 0.35) -> None:
        from ultralytics import YOLO

        self.model, self.conf = YOLO(str(weights)), conf

    def __call__(self, bgr: np.ndarray, gripper_mask: np.ndarray | None) -> list[tuple[np.ndarray, str | None, float]]:
        r = self.model.predict(bgr, conf=self.conf, retina_masks=True, verbose=False)[0]
        if r.masks is None:
            return []
        out = []
        for m, c, p in zip(r.masks.data.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            m = m > 0.5
            if m.shape != bgr.shape[:2]:
                m = cv2.resize(m.astype(np.uint8), bgr.shape[1::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            if gripper_mask is not None:
                m &= ~gripper_mask
            if m.sum() >= 300:
                out.append((m, r.names[int(c)], float(p)))
        return out


class VisionService:
    def __init__(self, frame_source: Callable[[], np.ndarray], H: np.ndarray, gripper_mask: np.ndarray,
                 segmenter, staging: list[tuple[float, float]], log_dir: Path | None = None,
                 staging_yaw_deg: float = 0.0) -> None:
        self.frame_source, self.H, self.gripper_mask, self.segmenter = frame_source, H, gripper_mask, segmenter
        self.staging, self.staging_yaw_deg = staging, staging_yaw_deg
        self.used_staging: set[int] = set()
        self.log_dir = log_dir
        self.instruments = load_instruments()
        self.templates = build_templates(self.instruments, TEMPLATE_MM_PER_PX)
        self.lock = threading.Lock()
        self.last_debug: np.ndarray | None = None

    def detect(self, bgr: np.ndarray, mask_gripper: bool = True) -> list[Detection]:
        segs = self.segmenter(bgr, self.gripper_mask if mask_gripper else None)
        return [identify(m, c, self.H, self.templates, self.instruments, seg_conf=p) for m, c, p in segs]

    def plan(self, target_id: str) -> dict:
        with self.lock:
            if target_id not in self.instruments:
                return {"action": "fail", "target": None, "place": None,
                        "say": f"I don't know what {target_id.replace('_', ' ')} looks like"}
            bgr = self.frame_source()
            dets = self.detect(bgr)
            staged_xy = [self.staging[i] for i in self.used_staging]
            p = plan(dets, target_id, self.H, self.staging, self.used_staging, staged_xy=staged_xy,
                     staging_yaw_deg=self.staging_yaw_deg)
            if p["action"] == "move_aside":
                self.used_staging.add(p["staging_index"])
            chosen = None
            if p.get("target"):
                chosen = min(dets, key=lambda d: np.hypot(d.grasp_mm[0] - p["target"]["x_mm"],
                                                          d.grasp_mm[1] - p["target"]["y_mm"]), default=None)
            self.last_debug = annotate(bgr, dets, chosen)
            self._log(bgr, {"request": target_id, "plan": p,
                            "detections": [{"fine_id": d.fine_id, "coarse": d.coarse, "candidates": d.candidates,
                                            "grasp_mm": d.grasp_mm, "exposure": d.exposure,
                                            "confidence": d.confidence, "length_mm": d.length_mm} for d in dets]})
            return p

    def count(self) -> dict:
        """Which instruments are in view, for the console's end-of-case count check.

        An instrument counts as `confident` when its identity was settled; one that could be several look-alikes
        (all scalpels, both micro scissors) is reported as its first candidate and the console treats the group as
        interchangeable. Blobs no template explains are only tallied as `unrecognised`."""
        with self.lock:
            bgr = self.frame_source()
            dets = self.detect(bgr)
            self.last_debug = annotate(bgr, dets, None)
            seen = [{"id": d.fine_id or d.candidates[0], "confident": d.fine_id is not None, "exposure": round(d.exposure, 3)}
                    for d in dets if d.candidates]
            return {"instruments": seen, "unrecognised": sum(1 for d in dets if not d.candidates)}

    def verify(self, expected: str) -> dict:
        """Classify the instrument hanging in the gripper (the blob that reaches into the gripper zone)."""
        with self.lock:
            bgr = self.frame_source()
            dets = self.detect(bgr, mask_gripper=False)
            zone = cv2.dilate(self.gripper_mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            held = [d for d in dets if (d.mask & zone).any()]
            self.last_debug = annotate(bgr, dets, held[0] if held else None)
            if not held:
                return {"ok": False, "detected": None, "confidence": 0.0}
            d = max(held, key=lambda d: (d.mask & zone).sum())
            want = self.instruments.get(expected)
            ok = want is not None and d.coarse == want.coarse
            detected = expected if ok else (d.fine_id or d.coarse)
            return {"ok": bool(ok), "detected": detected, "confidence": round(d.seg_conf if ok else 0.0, 3)}

    def _log(self, bgr: np.ndarray, record: dict) -> None:
        if self.log_dir is None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stem = self.log_dir / f"{time.time():.3f}"
        cv2.imwrite(str(stem) + ".jpg", bgr)
        Path(str(stem) + ".json").write_text(json.dumps(record, default=float))

    def debug_jpeg(self) -> bytes:
        img = self.last_debug if self.last_debug is not None else self.frame_source()
        return cv2.imencode(".jpg", img)[1].tobytes()


def make_server(svc: VisionService, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, data: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj, default=float).encode())

        def do_GET(self):
            if self.path.startswith("/debug"):
                return self._send(200, svc.debug_jpeg(), "image/jpeg")
            if self.path.startswith("/status"):
                return self._json(200, {"used_staging": sorted(svc.used_staging), "staging": svc.staging})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path.startswith("/plan"):
                    return self._json(200, svc.plan(str(body.get("instrument", ""))))
                if self.path.startswith("/verify"):
                    return self._json(200, svc.verify(str(body.get("expected", ""))))
                if self.path.startswith("/count"):
                    return self._json(200, svc.count())
                if self.path.startswith("/reset_staging"):
                    svc.used_staging.clear()
                    return self._json(200, {"ok": True})
                self._json(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001 - always answer; the arm turns errors into speech
                log.exception("request failed")
                self._json(200, {"action": "fail", "target": None, "place": None, "say": f"vision error: {e}"})

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def _gripper_mask(poly, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m.astype(bool)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="scrub_vision.service")
    ap.add_argument("--weights", help="YOLO-seg weights; omit for the HSV fallback segmenter")
    ap.add_argument("--frame", help="analyse this image instead of the camera (testing)")
    ap.add_argument("--camera-yaml", default=str(ROOT / "config" / "camera.yaml"))
    ap.add_argument("--arm-yaml", default=str(ROOT / "arm" / "config" / "arm.yaml"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    cam = yaml.safe_load(Path(args.camera_yaml).read_text())
    H = np.array(cam["H_px_to_mm"], float)
    pick = (yaml.safe_load(Path(args.arm_yaml).read_text()) or {}).get("pick", {})
    staging = [tuple(map(float, p)) for p in pick.get("staging_mm", [])]
    if args.frame:
        img = cv2.imread(args.frame)
        source = lambda: img  # noqa: E731
    else:
        from .camera import Camera

        K = np.array(cam["K"], float) if cam.get("K") else None
        dist = np.array(cam.get("dist") or [0] * 5, float)
        c = Camera(int(cam.get("camera_index", 0)), K, dist, cam.get("width"), cam.get("height"))
        source = c.latest
    shape = source().shape[:2]
    gm = _gripper_mask(cam.get("gripper_mask_poly_px") or
                       [[0, int(shape[0] * .75)], [shape[1], int(shape[0] * .75)], [shape[1], shape[0]], [0, shape[0]]],
                       shape)
    seg = YoloSegmenter(args.weights) if args.weights else BackgroundSegmenter()
    svc = VisionService(source, H, gm, seg, staging, log_dir=Path(__file__).resolve().parents[1] / "logs",
                        staging_yaw_deg=float(pick.get("staging_yaw_deg", 0.0)))
    httpd = make_server(svc, args.host, args.port)
    print(f"scrub_vision listening on http://{args.host}:{args.port} ({'YOLO' if args.weights else 'background-difference fallback'})",
          flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
