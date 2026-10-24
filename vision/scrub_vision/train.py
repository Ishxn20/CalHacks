"""Train the instrument detector on the validated dataset.

    python -m scrub_vision.train                 # data/clean, yolo11s, 30 epochs, GPU (cuda/mps) if present
    python -m scrub_vision.train --epochs 1 --fraction 0.05    # smoke test

Weights land in runs/<name>/weights/best.pt. After training it prints per-class precision/recall and the
most-confused class for each instrument (mAP alone hides the failure we care about: a #11 called a #20), then
packages everything worth sharing into results/<name>/ (results.json, best.pt, curves, confusion matrix).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SHARED = ("weights/best.pt", "results.csv", "args.yaml", "confusion_matrix_normalized.png", "results.png")


def confusions(matrix: np.ndarray, names: list[str]) -> list[dict]:
    """matrix[pred, true] (last row/col = background). One row per class: its most-confused other class."""
    nc, rows = len(names), []
    for t in range(nc):
        total = matrix[:, t].sum()
        if total == 0:
            continue
        rates = matrix[:nc, t] / total
        rates[t] = 0
        p = int(rates.argmax())
        rows.append({"true": names[t], "pred": names[p], "rate": round(float(rates[p]), 4)})
    return rows


def export_results(name: str, args, metrics, names, rows, seconds: float) -> Path:
    run, out = ROOT / "runs" / name, ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    for f in SHARED:
        if (run / f).exists():
            shutil.copy2(run / f, out / Path(f).name)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    csv_rows = (run / "results.csv").read_text().strip().splitlines() if (run / "results.csv").exists() else []
    data_root = Path(args.data).parent
    bad = [r for r in rows if r["rate"] > args.max_confusion]
    report = {
        "run": name, "git_commit": commit or None, "seconds": round(seconds),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else
                  ("mps" if torch.backends.mps.is_available() else "cpu"),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "model": args.model, "imgsz": args.imgsz, "batch": args.batch, "epochs_requested": args.epochs,
        "epochs_completed": max(len(csv_rows) - 1, 0),
        "dataset": {s: len(list((data_root / "images" / s).glob("*"))) for s in ("train", "val")},
        "map50": round(float(metrics.box.map50), 4), "map50_95": round(float(metrics.box.map), 4),
        "per_class": [{"class": n, "precision": round(float(metrics.box.p[i]), 4),
                       "recall": round(float(metrics.box.r[i]), 4), "map50": round(float(metrics.box.ap50[i]), 4)}
                      for i, n in enumerate(names)],
        "confusions": rows, "max_confusion": args.max_confusion,
        "confusion_pass": not bad, "confusion_failures": bad,
    }
    (out / "results.json").write_text(json.dumps(report, indent=2))
    return out


def main() -> None:
    t0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "clean" / "data.yaml")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fraction", type=float, default=1.0, help="use only this fraction of train images")
    ap.add_argument("--max-confusion", type=float, default=0.05)
    ap.add_argument("--name", default="tools")
    ap.add_argument("--resume", action="store_true", help="resume runs/<name>/weights/last.pt")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("0" if torch.cuda.is_available() else "cpu")
    if args.resume:  # continue an interrupted run (battery died, Ctrl-C) from its last epoch; settings come from the saved run
        YOLO(ROOT / "runs" / args.name / "weights" / "last.pt").train(resume=True)
    else:
        # No flips: a mirrored Potts/micro scissors is not a real tool. Renders already cover every rotation.
        # patience=0: never early-stop, overfit to these tools on purpose.
        YOLO(args.model).train(data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                               device=device, fraction=args.fraction, project=str(ROOT / "runs"), name=args.name,
                               exist_ok=True, fliplr=0.0, flipud=0.0, patience=0, workers=args.workers, plots=True)
    metrics = YOLO(ROOT / "runs" / args.name / "weights" / "best.pt").val(
        data=str(args.data), imgsz=args.imgsz, device=device, plots=True)

    names = [metrics.names[i] for i in sorted(metrics.names)]
    print(f"\nmAP50 {metrics.box.map50:.3f}   mAP50-95 {metrics.box.map:.3f}")
    print("per class (precision / recall / mAP50):")
    for i, n in enumerate(names):
        print(f"  {n:28s} {metrics.box.p[i]:.3f} / {metrics.box.r[i]:.3f} / {metrics.box.ap50[i]:.3f}")
    rows = confusions(metrics.confusion_matrix.matrix, names)
    print("\nconfusions (most-confused other class):")
    for r in rows:
        print(f"  {r['true']:28s} -> {r['pred']:28s} {r['rate']:6.1%}")
    bad = [r for r in rows if r["rate"] > args.max_confusion]
    print("\nFAIL, confusable classes above limit:" if bad else "\nOK: no class confused above the limit.")
    for r in bad:
        print(f"  {r['true']} -> {r['pred']} {r['rate']:.1%}")
    print(f"\nshareable results: {export_results(args.name, args, metrics, names, rows, time.time() - t0)}")


if __name__ == "__main__":
    main()
