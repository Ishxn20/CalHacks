"""Synthetic pile dataset in Ultralytics YOLO-seg format.

    python -m scrub_vision.synth --n 3000 --out data/synth [--source random|mujoco]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import yaml

from .render import CamSpec, add_gripper, pose, render_scene, scene_mesh
from .templates import Instrument, classes, fill_tris, load_instruments

PILE_CENTER_MM = (0.0, -40.0)     # above the gripper, which occupies the bottom of the frame
PILE_HALF_MM = (120.0, 80.0)
HM_RES = 2.0                      # mm per heightmap cell
MIN_AREA_PX = 150

Pile = list[tuple[str, np.ndarray]]   # (fine_id, 4x4 pose in mm)


def _footprint(mesh, T, shape, origin) -> np.ndarray:
    v = mesh.vertices @ T[:3, :3].T + T[:3, 3]
    xy = (v[:, :2] - origin) / HM_RES
    fp = np.zeros(shape, np.uint8)
    fill_tris(fp, xy[mesh.faces], 1)
    return fp.astype(bool)


def random_pile(instruments: dict[str, Instrument], rng: np.random.Generator, n_items: int) -> Pile:
    """Drop n instruments one after another; each rests on the highest thing under its footprint."""
    ids = rng.choice(list(instruments), size=n_items, replace=n_items > len(instruments))
    cx, cy = PILE_CENTER_MM
    hx, hy = PILE_HALF_MM
    origin = np.array([cx - hx - 150, cy - hy - 150])
    shape = (int((2 * hy + 300) / HM_RES), int((2 * hx + 300) / HM_RES))
    height = np.zeros(shape, np.float32)
    pile: Pile = []
    for fid in ids:
        mesh = scene_mesh(instruments[fid])
        t = mesh.extents[2]
        x, y = rng.uniform(cx - hx, cx + hx), rng.uniform(cy - hy, cy + hy)
        T = pose(x, y, 0.0, rng.uniform(-np.pi, np.pi), flip=bool(rng.random() < 0.5))
        fp = _footprint(mesh, T, shape, origin)
        z0 = float(height[fp].max()) if fp.any() else 0.0
        T[2, 3] = z0 + t / 2
        height[fp] = z0 + t
        pile.append((str(fid), T))
    return pile


def _polygon(mask: np.ndarray) -> np.ndarray | None:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_AREA_PX:
        return None
    c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
    return c if len(c) >= 3 else None


def write_yolo_labels(lab_path: Path, shape: tuple[int, int], masks: list[np.ndarray], class_ids: list[int]) -> int:
    h, w = shape
    lines = []
    for m, c in zip(masks, class_ids):
        poly = _polygon(m)
        if poly is None:
            continue
        coords = (poly / [w, h]).clip(0, 1).ravel()
        lines.append(f"{c} " + " ".join(f"{x:.5f}" for x in coords))
    lab_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def write_yolo_sample(img_path: Path, lab_path: Path, bgr: np.ndarray, masks: list[np.ndarray],
                      class_ids: list[int]) -> int:
    cv2.imwrite(str(img_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, int(np.random.default_rng().integers(70, 96))])
    return write_yolo_labels(lab_path, bgr.shape[:2], masks, class_ids)


def make_dataset(out: Path | str, n: int, seed: int = 0, val_frac: float = 0.1,
                 poses_source: Callable[[np.random.Generator], Pile] | None = None, prefix: str = "") -> Path:
    out = Path(out)
    insts = load_instruments()
    names = classes()
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    for i in range(n):
        pile = poses_source(rng) if poses_source else random_pile(insts, rng, int(rng.integers(3, 9)))
        cam = CamSpec(height_mm=400.0 * rng.uniform(0.9, 1.1))
        items = [(scene_mesh(insts[fid]), T) for fid, T in pile]
        bgr, masks = render_scene(items, cam, rng)
        if rng.random() < 0.7:
            bgr, gm = add_gripper(bgr, rng)
            masks = [m & ~gm for m in masks]
        if rng.random() < 0.5:
            bgr = cv2.GaussianBlur(bgr, (0, 0), rng.uniform(0.3, 1.2))
        bgr = np.clip(bgr + rng.normal(0, rng.uniform(1, 5), bgr.shape), 0, 255).astype(np.uint8)
        split = "val" if i < n_val else "train"
        stem = f"{prefix}{i:05d}"
        write_yolo_sample(out / "images" / split / f"{stem}.jpg", out / "labels" / split / f"{stem}.txt",
                          bgr, masks, [names.index(insts[fid].coarse) for fid, _ in pile])
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": dict(enumerate(names))},
        sort_keys=False))
    return out / "data.yaml"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="scrub_vision.synth")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["random", "mujoco"], default="random")
    args = ap.parse_args(argv)
    src = None
    if args.source == "mujoco":
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # repo root, for the `sim` package
        from sim.pile_sim import poses_source as src   # Track C
    print(make_dataset(args.out, args.n, args.seed, poses_source=src, prefix=args.source[0]))


if __name__ == "__main__":
    main()
