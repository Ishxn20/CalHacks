"""Train / fine-tune the pile segmentation model.

    python -m scrub_vision.train --data data/synth/data.yaml data/synth_mj/data.yaml --epochs 30
    python -m scrub_vision.train --data data/mixed.yaml --model runs/pile/weights/best.pt --epochs 10   # + real
Copies the best weights to weights/pile_seg.pt.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]


def combine(datas: list[Path], out: Path) -> Path:
    """One data.yaml whose train/val are the union of several YOLO datasets (same class list)."""
    ds = [yaml.safe_load(Path(d).read_text()) for d in datas]
    names = ds[0]["names"]
    if any(d["names"] != names for d in ds):
        raise ValueError("datasets have different class lists")
    root = lambda d: Path(d["path"])  # noqa: E731
    out.write_text(yaml.safe_dump({"train": [str(root(d) / d["train"]) for d in ds],
                                   "val": [str(root(d) / d["val"]) for d in ds], "names": names}, sort_keys=False))
    return out


def train(datas: list[Path], model: str = "yolo11n-seg.pt", epochs: int = 30, imgsz: int = 640,
          device: str = "mps", batch: int = 16, project: Path = PKG / "runs", name: str = "pile") -> Path:
    from ultralytics import YOLO

    data = Path(datas[0]) if len(datas) == 1 else combine(datas, Path(project).parent / f"{name}_data.yaml")
    m = YOLO(model)
    # Geometry augmentations only: hue jitter would teach the model that green isn't the instrument colour.
    r = m.train(data=str(data), epochs=epochs, imgsz=imgsz, device=device, batch=batch, project=str(project),
                name=name, exist_ok=True, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, degrees=180, flipud=0.5, fliplr=0.5,
                mosaic=0.5, plots=True, verbose=False)
    return Path(r.save_dir) / "weights" / "best.pt"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="scrub_vision.train")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--model", default="yolo11n-seg.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--name", default="pile")
    args = ap.parse_args(argv)
    best = train([Path(d) for d in args.data], args.model, args.epochs, args.imgsz, args.device, args.batch,
                 name=args.name)
    (PKG / "weights").mkdir(exist_ok=True)
    shutil.copy(best, PKG / "weights" / "pile_seg.pt")
    print(f"best weights: {best} -> {PKG / 'weights' / 'pile_seg.pt'}")


if __name__ == "__main__":
    main()
