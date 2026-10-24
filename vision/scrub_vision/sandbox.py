"""Sandbox identity model: photos of the real objects -> auto-labels -> 2-class YOLO.

    python -m scrub_vision.sandbox capture --camera 0 --name sandbox_plug_cover   # one object alone on the plain tray
    python -m scrub_vision.sandbox capture --camera 0 --name sandbox_caliper
    python -m scrub_vision.sandbox build                                          # label with the classical blob box
    python -m scrub_vision.sandbox train                                          # -> runs/sandbox/weights/best.pt

Capture from the same camera pose the arm uses (`tray_view`). Move and rotate the object a bit between shots:
press space to save a frame, or pass --every 0.7 to save automatically. Only frames with exactly one blob are kept,
because the label is that blob's box. Class names must equal instrument ids in config/tray.yaml.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import cv2

from .grasp import load_roi, segment_in
from .markers import without_markers

DATA = Path(__file__).resolve().parents[1] / "data" / "sandbox"
RAW = DATA / "raw"


def capture(camera: int, name: str, every: float | None) -> None:
    from .camera import Camera
    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob("*.jpg")))
    cam = Camera(camera)
    roi = load_roi()
    last = 0.0
    try:
        while True:
            frame = cam.frame()
            blobs = without_markers(segment_in(frame, roi), frame)
            ok = len(blobs) == 1
            view = frame.copy()
            cv2.putText(view, f"{name}: {n} saved, blobs={len(blobs)} {'OK' if ok else 'need exactly 1'}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if ok else (0, 0, 255), 2)
            cv2.imshow("capture (space=save, q=quit)", view)
            key = cv2.waitKey(1) & 0xFF
            auto = every is not None and time.time() - last > every
            if ok and (key == ord(" ") or auto):
                cv2.imwrite(str(out / f"{n:04d}.jpg"), frame)
                n, last = n + 1, time.time()
            if key == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


def build(val_every: int = 7) -> dict[str, int]:
    names = sorted(p.name for p in RAW.iterdir() if p.is_dir()) if RAW.exists() else []
    if not names:
        raise SystemExit(f"no photos in {RAW}; run `capture` first")
    for sub in ("images", "labels"):
        shutil.rmtree(DATA / sub, ignore_errors=True)
    counts = {n: 0 for n in names}
    roi = load_roi()
    for cls, name in enumerate(names):
        for i, jpg in enumerate(sorted((RAW / name).glob("*.jpg"))):
            img = cv2.imread(str(jpg))
            blobs = without_markers(segment_in(img, roi), img)
            if len(blobs) != 1:
                continue   # lighting changed or something else is in view; skip rather than mislabel
            h, w = img.shape[:2]
            xs, ys = zip(*blobs[0].box_px)
            x1, x2, y1, y2 = max(min(xs), 0), min(max(xs), w), max(min(ys), 0), min(max(ys), h)
            split = "val" if i % val_every == 0 else "train"
            stem = f"{name}_{jpg.stem}"
            (DATA / "images" / split).mkdir(parents=True, exist_ok=True)
            (DATA / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy(jpg, DATA / "images" / split / f"{stem}.jpg")
            (DATA / "labels" / split / f"{stem}.txt").write_text(
                f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
            counts[name] += 1
    lines = [f"path: {DATA}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    (DATA / "data.yaml").write_text("\n".join(lines) + "\n")
    print(f"labelled {counts} -> {DATA / 'data.yaml'}")
    return counts


def train(epochs: int, base: str) -> None:
    import torch
    from ultralytics import YOLO
    runs = DATA.parents[1] / "runs"
    YOLO(base).train(data=str(DATA / "data.yaml"), epochs=epochs, imgsz=512, project=str(runs), name="sandbox",
                     exist_ok=True, device="mps" if torch.backends.mps.is_available() else "cpu")
    print(f"weights: {runs / 'sandbox' / 'weights' / 'best.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--camera", type=int, required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--every", type=float)
    sub.add_parser("build")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--base", default=str(Path(__file__).resolve().parents[1] / "yolo11n.pt"))
    a = ap.parse_args()
    {"capture": lambda: capture(a.camera, a.name, a.every), "build": build,
     "train": lambda: train(a.epochs, a.base)}[a.cmd]()


if __name__ == "__main__":
    main()
