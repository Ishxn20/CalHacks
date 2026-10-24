"""Validate a YOLO dataset before training. Never modifies the source; builds a clean symlink copy.

    python -m scrub_vision.check_data                       # data/synth -> data/clean
    python -m scrub_vision.check_data --src path/to/other/dataset

Rejects images that fail to decode, are tiny/blank/blurry, are duplicates (also across train/val, so val
can't leak from train), or whose labels are missing, malformed, out of range, degenerate or duplicated.
Writes rejected.csv (one reason per file) and prints per-class counts before and after.
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from scrub_arm.config import load_tray_config

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "val")
POP = np.array([bin(i).count("1") for i in range(256)], np.uint8)


def load_names(src: Path) -> list[str]:
    names = yaml.safe_load((src / "data.yaml").read_text())["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    expected = list(load_tray_config().instruments)
    if names != expected:  # same order too: class index is the contract with the model
        raise SystemExit(f"dataset classes != config/tray.yaml instruments\n  dataset: {names}\n  tray:    {expected}")
    return names


def check_labels(text: str, n_classes: int, min_box: float) -> tuple[list[tuple], str | None]:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        try:
            if len(parts) != 5 or int(parts[0]) != float(parts[0]):
                return [], "malformed label line"
            c, x, y, w, h = int(parts[0]), *map(float, parts[1:])
        except ValueError:
            return [], "malformed label line"
        if not 0 <= c < n_classes:
            return [], "class id out of range"
        if not all(np.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
            return [], "non-positive box"
        if x - w / 2 < -1e-4 or y - h / 2 < -1e-4 or x + w / 2 > 1 + 1e-4 or y + h / 2 > 1 + 1e-4:
            return [], "box outside image"
        if min(w, h) < min_box:
            return [], "box too small"
        rows.append((c, x, y, w, h))
    if not rows:
        return [], "empty label file"
    for i, a in enumerate(rows):
        for b in rows[:i]:
            if a[0] == b[0] and _iou(a[1:], b[1:]) > 0.95:
                return [], "duplicate box"
    return rows, None


def _iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-12)


def check_pair(args) -> tuple[str, str | None, np.ndarray | None]:
    img_path, lbl_path, n_classes, cfg = args
    img = cv2.imread(img_path)
    if img is None:
        return img_path, "image does not decode", None
    h, w = img.shape[:2]
    if min(h, w) < cfg["min_size"]:
        return img_path, "image too small", None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.std() < cfg["min_std"]:
        return img_path, "image blank/flat", None
    if cv2.Laplacian(gray, cv2.CV_64F).var() < cfg["min_sharp"]:
        return img_path, "image too blurry", None
    if not Path(lbl_path).exists():
        return img_path, "missing label file", None
    rows, err = check_labels(Path(lbl_path).read_text(), n_classes, cfg["min_box"])
    if err:
        return img_path, err, None
    for _, x, y, bw, bh in rows:  # a box over flat background means the label is wrong or the tool is invisible
        x0, x1, y0, y1 = int((x - bw / 2) * w), int((x + bw / 2) * w) + 1, int((y - bh / 2) * h), int((y + bh / 2) * h) + 1
        if gray[y0:y1, x0:x1].std() < cfg["min_box_std"]:
            return img_path, "box over flat region", None
    bits = cv2.resize(gray, (17, 16), interpolation=cv2.INTER_AREA)
    return img_path, None, np.packbits(bits[:, 1:] > bits[:, :-1])  # 256-bit dHash


def find_duplicates(hashes: dict[str, np.ndarray], max_dist: int) -> dict[str, str]:
    """path -> reason, for the later of each near-identical pair. Train is scanned first, so val dups lose."""
    paths = list(hashes)
    mat = np.stack([hashes[p] for p in paths])
    out: dict[str, str] = {}
    for i, p in enumerate(paths):
        if p in out or i == 0:
            continue
        dist = POP[np.bitwise_xor(mat[:i], mat[i])].sum(1)
        for j in np.nonzero(dist <= max_dist)[0]:
            if paths[j] not in out:
                out[p] = f"near-duplicate of {Path(paths[j]).name}"
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "synth")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "clean")
    ap.add_argument("--min-size", type=int, default=320)
    ap.add_argument("--min-std", type=float, default=6.0, help="min grayscale std (blank image)")
    ap.add_argument("--min-sharp", type=float, default=3.0, help="min Laplacian variance (blur); 15 also drops mildly soft images worth training on")
    ap.add_argument("--min-box", type=float, default=0.01, help="min box side, fraction of image")
    ap.add_argument("--min-box-std", type=float, default=4.0, help="min grayscale std inside a box")
    ap.add_argument("--dup-dist", type=int, default=6, help="max dHash bit distance (of 256) to call a duplicate")
    args = ap.parse_args()

    names = load_names(args.src)
    cfg = {"min_size": args.min_size, "min_std": args.min_std, "min_sharp": args.min_sharp,
           "min_box": args.min_box, "min_box_std": args.min_box_std}
    jobs = [(str(p), str(args.src / "labels" / s / (p.stem + ".txt")), len(names), cfg)
            for s in SPLITS for p in sorted((args.src / "images" / s).glob("*.[jp][pn]g"))]
    with mp.Pool() as pool:
        results = pool.map(check_pair, jobs, chunksize=32)

    rejected = {p: why for p, why, _ in results if why}
    dups = find_duplicates({p: h for p, why, h in results if not why}, args.dup_dist)
    rejected |= dups

    if args.out.exists():
        shutil.rmtree(args.out)
    kept = Counter(), Counter()
    for sub in ("images", "labels"):
        for s in SPLITS:
            (args.out / sub / s).mkdir(parents=True)
    before, after = {s: Counter() for s in SPLITS}, {s: Counter() for s in SPLITS}
    for (img, lbl, _, _), (_, why, _) in zip(jobs, results):
        split = Path(img).parent.name
        classes = [int(l.split()[0]) for l in Path(lbl).read_text().splitlines()] if Path(lbl).exists() else []
        before[split].update(classes)
        if img in rejected:
            continue
        after[split].update(classes)
        (args.out / "images" / split / Path(img).name).symlink_to(Path(img).resolve())
        (args.out / "labels" / split / Path(lbl).name).symlink_to(Path(lbl).resolve())
    (args.out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(args.out.resolve()), "train": "images/train", "val": "images/val", "names": dict(enumerate(names))}))
    with open(args.out / "rejected.csv", "w", newline="") as f:
        csv.writer(f).writerows([("image", "reason"), *sorted(rejected.items())])

    print(f"{len(jobs)} images checked, {len(jobs) - len(rejected)} kept, {len(rejected)} rejected")
    for reason, n in Counter(r.split(" of ")[0] if r.startswith("near-dup") else r for r in rejected.values()).most_common():
        print(f"  {n:5d}  {reason}")
    print("\nlabeled boxes per class (train kept/total, val kept/total):")
    for i, name in enumerate(names):
        print(f"  {name:28s} {after['train'][i]:5d}/{before['train'][i]:<5d} {after['val'][i]:5d}/{before['val'][i]:<5d}")
    thin = [n for i, n in enumerate(names) if after["train"][i] < 100 or after["val"][i] < 20]
    if thin:
        raise SystemExit(f"too few clean examples for: {thin}")
    print(f"\nclean dataset: {args.out / 'data.yaml'}  (details in rejected.csv)")


if __name__ == "__main__":
    main()
