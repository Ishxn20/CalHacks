"""Perception service for the arm.

    python -m scrub_vision.service [--weights weights/pile_seg.pt] [--frame some.jpg] [--port 8770]

    POST /plan {"instrument": id}            -> grasp plan, or why not (see planner.py)
    POST /verify {"expected": id, "slot": _} -> {"ok", "detected", "confidence"} (arm's verify hook)
    POST /observe {"q": joints rad, "reset"} -> red parts: add this camera view; {"seen", "incomplete"} (the arm moves
                                                the wrist camera until no part is cut off, then asks /plan)
    POST /reset_staging                      -> a new pile: forget what was learned about this one
    GET  /debug                              -> annotated JPEG of the last analysed frame
    GET  /status
Reads repo-root config/camera.yaml (calibration).
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

from .identify import Detection, _apply, annotate, identify, segment_background
from .planner import plan
from .templates import build_templates, load_instruments

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
KNOWN_RADIUS_MM = 40.0   # a marked identity applies to the detection whose grasp point is this close
TEMPLATE_MM_PER_PX = 0.5


class BackgroundSegmenter:
    """Fallback: anything that isn't table surface, any colour; no class (identify() decides).
    Touching instruments merge, so it only works on spread-out instruments."""

    def __call__(self, bgr: np.ndarray, gripper_mask: np.ndarray | None) -> list[tuple[np.ndarray, str | None, float]]:
        return [(m, None, 1.0) for m in segment_background(bgr, gripper_mask=gripper_mask)]


class YoloSegmenter:
    # imgsz 960 on the real 1920x1080 wrist camera: at the trained 640 the scalpels were missed; 960 found all four
    # printed tools on the first real scan frame (0.73-0.90) with nothing else on the mat.
    def __init__(self, weights: str | Path, conf: float = 0.35, imgsz: int = 960) -> None:
        from ultralytics import YOLO

        self.model, self.conf, self.imgsz = YOLO(str(weights)), conf, imgsz

    def __call__(self, bgr: np.ndarray, gripper_mask: np.ndarray | None) -> list[tuple[np.ndarray, str | None, float]]:
        r = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, retina_masks=True, verbose=False)[0]
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


class RedPartsSegmenter:
    """Our red printed parts on white paper, identified by measured shape (red_parts.py), at the scan pose."""

    def __init__(self) -> None:
        from .red_parts import RedPartsDetector
        self.det = RedPartsDetector()
        self.q = np.array(self.det.model["scan_q"], float)
        self.views: list[Detection] | None = None   # merged detections from /observe (None: use the scan frame)

    def H(self, shape) -> np.ndarray:
        return self.det.cam.H_px_to_mm(self.q, shape)

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        return self.views if self.views is not None else self.det.detect(bgr, self.q)

    def observe(self, bgr: np.ndarray, q, reset: bool) -> dict:
        """One more camera view (taken at joints q); keeps each part's best view so far."""
        dets = self.det.detect(bgr, np.asarray(q, float))
        self.views = self.det.merge([] if reset or self.views is None else self.views, dets, self.q, bgr.shape)
        parts = [d for d in self.views if d.candidates]
        return {"seen": len(parts), "incomplete": sum(d.exposure < 1 for d in parts)}


class VisionService:
    def __init__(self, frame_source: Callable[[], np.ndarray], H: np.ndarray, gripper_mask: np.ndarray,
                 segmenter, log_dir: Path | None = None, region: np.ndarray | None = None) -> None:
        self.frame_source, self.H, self.gripper_mask, self.segmenter = frame_source, H, gripper_mask, segmenter
        self.region = region          # work mat in pixels; detections mostly off it are ignored
        self.known: list[tuple[float, float, str]] = []   # (x_mm, y_mm, fine_id) confirmed close up (mark)
        self.log_dir = log_dir
        self.instruments = load_instruments()
        self.templates = build_templates(self.instruments, TEMPLATE_MM_PER_PX)
        self.lock = threading.Lock()
        self.last_debug: np.ndarray | None = None

    def detect(self, bgr: np.ndarray, mask_gripper: bool = True) -> list[Detection]:
        if isinstance(self.segmenter, RedPartsSegmenter):
            return self.segmenter.detect(bgr)
        segs = self.segmenter(bgr, self.gripper_mask if mask_gripper else None)
        if self.region is not None:
            segs = [s for s in segs if (s[0] & self.region).sum() >= 0.8 * s[0].sum()]
        dets = [identify(m, c, self.H, self.templates, self.instruments, seg_conf=p) for m, c, p in segs]
        for d in dets:                # what the arm saw up close beats what the scan could tell (e.g. the scissors' hook)
            for x, y, fid in self.known:
                if np.hypot(d.grasp_mm[0] - x, d.grasp_mm[1] - y) < KNOWN_RADIUS_MM \
                        and d.coarse == self.instruments[fid].coarse:
                    d.fine_id, d.alt_ids, d.candidates = fid, [], [fid]
        return dets

    def mark(self, x_mm: float, y_mm: float, fine_id: str) -> None:
        """The instrument at (x, y) is fine_id (seen from the hover); later plans treat it as that."""
        with self.lock:
            self.known = [k for k in self.known if np.hypot(k[0] - x_mm, k[1] - y_mm) >= KNOWN_RADIUS_MM]
            self.known.append((float(x_mm), float(y_mm), fine_id))

    def observe(self, q, reset: bool = False) -> dict:
        with self.lock:
            if not isinstance(self.segmenter, RedPartsSegmenter):
                return {"seen": 0, "incomplete": 0}
            return self.segmenter.observe(self.frame_source(), q, reset)

    def plan(self, target_id: str) -> dict:
        with self.lock:
            if target_id not in self.instruments:
                return {"action": "fail", "target": None, "place": None,
                        "say": f"I don't know what {target_id.replace('_', ' ')} looks like"}
            bgr = self.frame_source()
            dets = self.detect(bgr)
            p = plan(dets, target_id, self.H)
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
            self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path.startswith("/plan"):
                    return self._json(200, svc.plan(str(body.get("instrument", ""))))
                if self.path.startswith("/verify"):
                    return self._json(200, svc.verify(str(body.get("expected", ""))))
                if self.path.startswith("/observe"):
                    return self._json(200, svc.observe(body["q"], bool(body.get("reset"))))
                if self.path.startswith("/reset_staging"):
                    svc.known.clear()                 # a new pile: forget what was seen up close
                    if isinstance(svc.segmenter, RedPartsSegmenter):
                        svc.segmenter.views = None
                    return self._json(200, {"ok": True})
                self._json(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001 - always answer; the arm turns errors into speech
                log.exception("request failed")
                self._json(200, {"action": "fail", "target": None, "place": None, "say": f"vision error: {e}"})

    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd


def region_mask(poly_mm, H: np.ndarray, shape) -> np.ndarray:
    """Table polygon (mm, e.g. the work mat's corners) -> pixel mask through the inverse homography."""
    return _gripper_mask(_apply(np.linalg.inv(H), np.asarray(poly_mm, float)), shape)


def _gripper_mask(poly, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m.astype(bool)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="scrub_vision.service")
    ap.add_argument("--weights", help="YOLO-seg weights; omit for the HSV fallback segmenter")
    ap.add_argument("--red-parts", action="store_true", help="our 8 red printed parts (red_parts.py); ignores --weights")
    ap.add_argument("--frame", help="analyse this image instead of the camera (testing)")
    ap.add_argument("--camera-yaml", default=str(ROOT / "config" / "camera.yaml"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    cam = yaml.safe_load(Path(args.camera_yaml).read_text())
    H = np.array(cam["H_px_to_mm"], float)
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
    seg = RedPartsSegmenter() if args.red_parts else YoloSegmenter(args.weights) if args.weights else BackgroundSegmenter()
    if args.red_parts:
        H = seg.H(shape)               # its own camera model (FK + hand-eye), the frame its grasp points are in
    svc = VisionService(source, H, gm, seg, log_dir=Path(__file__).resolve().parents[1] / "logs",
                        region=region_mask(cam["mat_mm"], H, shape) if cam.get("mat_mm") and not args.red_parts else None)
    httpd = make_server(svc, args.host, args.port)
    print(f"scrub_vision listening on http://{args.host}:{args.port} ({'red parts' if args.red_parts else 'YOLO' if args.weights else 'background-difference fallback'})",
          flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
