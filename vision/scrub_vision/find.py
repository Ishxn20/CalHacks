"""Find the requested instrument in the tray camera image and write its crop + a JSON result.

    python -m scrub_vision.find                     # live: watch ../last_command.json, grab camera frames per command
    python -m scrub_vision.find --image tray.jpg    # still image, uses ../last_command.json (or --instrument scalpel_20)

The detector knows every instrument in config/tray.yaml. We keep only detections of the *requested* class and take
the most confident of those, so a #11 sitting next to the #20 can never be returned for a #20 request. If nothing
for that class clears --min-conf the result is found=false: no result beats the wrong tool.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
from scrub_arm.config import load_tray_config

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
CMD_PATH = REPO / "last_command.json"
OUT_JSON, OUT_CROP, OUT_ANN = REPO / "last_detection.json", REPO / "last_crop.jpg", REPO / "last_annotated.jpg"


def select(dets: list[tuple[int, float, tuple]], class_id: int, min_conf: float):
    """dets are (class_id, conf, (x1, y1, x2, y2)). Best detection of class_id at or above min_conf, else None."""
    best = [d for d in dets if d[0] == class_id and d[1] >= min_conf]
    return max(best, key=lambda d: d[1]) if best else None


def detect(model, frame) -> list[tuple[int, float, tuple]]:
    r = model(frame, verbose=False, conf=0.05)[0]  # low floor; select() applies the real threshold
    return [(int(c), float(p), tuple(float(v) for v in xy)) for c, p, xy in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]


def write_result(instrument, best, frame, heard=None, error=None) -> dict:
    out = {"found": best is not None, "instrument": instrument, "heard": heard, "confidence": None,
           "bbox_xyxy": None, "crop": None, "annotated": None, "error": error, "ts": time.time()}
    if best is not None and frame is not None:
        _, conf, (x1, y1, x2, y2) = best
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        ann = frame.copy()
        cv2.rectangle(ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(ann, f"{instrument} {conf:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(str(OUT_CROP), frame[max(y1, 0):y2, max(x1, 0):x2])
        cv2.imwrite(str(OUT_ANN), ann)
        out.update(confidence=round(conf, 4), bbox_xyxy=[x1, y1, x2, y2], crop=OUT_CROP.name, annotated=OUT_ANN.name)
    fd, tmp = tempfile.mkstemp(dir=REPO, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, OUT_JSON)  # readers never see a half-written file
    return out


def read_command(path: Path = CMD_PATH) -> dict:
    return json.loads(path.read_text())


def load_model(weights: Path):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)]
    if names != list(load_tray_config().instruments):
        raise SystemExit(f"model classes != config/tray.yaml instruments; retrain.\n  model: {names}")
    return model, names


def best_over_frames(model, names, instrument, grab, window_s, min_conf):
    """Run the detector on frames from grab() for window_s seconds; keep the most confident hit.

    Slow cameras just yield fewer frames in the window; we never need a particular frame rate.
    """
    cid, best, best_frame, end = names.index(instrument), None, None, time.time() + window_s
    while True:  # always at least one frame, even with window_s=0 (still images)
        frame = grab()
        if frame is not None:
            hit = select(detect(model, frame), cid, min_conf)
            if hit and (best is None or hit[1] > best[1]):
                best, best_frame = hit, frame
            if best_frame is None:
                best_frame = frame
        if time.time() >= end and best_frame is not None:
            return best, best_frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, default=ROOT / "runs" / "tools" / "weights" / "best.pt")
    ap.add_argument("--image", type=Path, help="still image instead of the camera")
    ap.add_argument("--instrument", help="instrument id to find (default: from last_command.json)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--window", type=float, default=2.0, help="seconds of live frames to search per command")
    ap.add_argument("--min-conf", type=float, default=0.5)
    args = ap.parse_args()
    model, names = load_model(args.weights)

    def run(instrument, heard, grab):
        if instrument not in names:
            return write_result(instrument, None, None, heard, f"unknown instrument {instrument!r}")
        best, frame = best_over_frames(model, names, instrument, grab, 0 if args.image else args.window, args.min_conf)
        return write_result(instrument, best, frame, heard, None if best else "not found with enough confidence")

    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            raise SystemExit(f"cannot read {args.image}")
        cmd = {"instrument": args.instrument} if args.instrument else read_command()
        if cmd.get("instrument"):
            out = run(cmd["instrument"], cmd.get("heard"), lambda: img)
        else:
            out = write_result(None, None, None, cmd.get("heard"), cmd.get("error") or "no instrument in command")
        print(json.dumps(out, indent=2))
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera} (macOS: allow camera access for this terminal); try --image")
    seen = 0.0
    print("watching last_command.json (Ctrl-C to stop)")
    while True:
        if CMD_PATH.exists() and CMD_PATH.stat().st_mtime > seen:
            seen = CMD_PATH.stat().st_mtime
            cmd = read_command()
            if cmd.get("ok"):
                out = run(cmd["instrument"], cmd.get("heard"), lambda: cap.read()[1])
            else:
                out = write_result(None, None, None, cmd.get("heard"), cmd.get("error"))
            print(json.dumps(out))
        time.sleep(0.2)


if __name__ == "__main__":
    main()
