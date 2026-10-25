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

from .render import CamSpec, add_gripper, pose, random_filament_bgr, render_scene, scene_mesh
from .templates import Instrument, classes, fill_tris, load_instruments

PILE_CENTER_MM = (0.0, -40.0)     # above the gripper, which occupies the bottom of the frame
PILE_HALF_MM = (120.0, 80.0)
HM_RES = 2.0                      # mm per heightmap cell
MIN_AREA_PX = 150

Pile = list[tuple[str, np.ndarray]]   # (fine_id, 4x4 pose in mm)
Item = tuple[object, np.ndarray, str, object]   # (mesh, 4x4 pose in mm, class name, colour or None = the set's)
# Real wrist-camera frames of the bare work mat / table (no instruments in view): backgrounds for half the scenes
BG_DIR = Path(__file__).resolve().parents[2] / "hw_check" / "aim"


def _footprint(mesh, T, shape, origin) -> np.ndarray:
    v = mesh.vertices @ T[:3, :3].T + T[:3, 3]
    xy = (v[:, :2] - origin) / HM_RES
    fp = np.zeros(shape, np.uint8)
    fill_tris(fp, xy[mesh.faces], 1)
    return fp.astype(bool)


def random_distractor(rng: np.random.Generator):
    """A random non-instrument object (class `other`): box, rod, ring, flat irregular shape or a standing
    cylinder (bottle-like), in mm, centred, resting on its base."""
    import trimesh
    k = rng.integers(0, 5)
    if k == 0:
        m = trimesh.creation.box(rng.uniform([10, 5, 3], [90, 45, 30]))
    elif k == 1:
        m = trimesh.creation.cylinder(rng.uniform(2, 8), rng.uniform(40, 200), sections=16)
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))     # lying down
    elif k == 2:
        r = rng.uniform(8, 30)
        m = trimesh.creation.annulus(r * rng.uniform(0.5, 0.85), r, rng.uniform(2, 8), sections=24)
    elif k == 3:                                                # flat irregular (convex) slab
        pts = rng.normal(0, 1, (int(rng.integers(5, 12)), 2)) * rng.uniform([10, 4], [60, 30])
        h = rng.uniform(2, 15)
        m = trimesh.convex.convex_hull(np.r_[np.c_[pts, np.zeros(len(pts))], np.c_[pts, np.full(len(pts), h)]])
    else:
        m = trimesh.creation.cylinder(rng.uniform(15, 40), rng.uniform(30, 120), sections=24)   # standing
    m.vertices -= m.bounds.mean(0)
    return m


def random_pile(instruments: dict[str, Instrument], rng: np.random.Generator, n_items: int,
                n_other: int = 0) -> list[Item]:
    """Drop n instruments (each lying in one of its rests) and n_other random objects one after another; each
    rests on the highest thing under its footprint."""
    ids = list(rng.choice(list(instruments), size=n_items, replace=n_items > len(instruments))) + [None] * n_other
    rng.shuffle(ids)
    cx, cy = PILE_CENTER_MM
    hx, hy = PILE_HALF_MM
    origin = np.array([cx - hx - 150, cy - hy - 150])
    shape = (int((2 * hy + 300) / HM_RES), int((2 * hx + 300) / HM_RES))
    height = np.zeros(shape, np.float32)
    pile: list[Item] = []
    for fid in ids:
        if fid is None:
            mesh, cls = random_distractor(rng), "other"
            colour = random_filament_bgr(rng) if rng.random() < 0.7 else None     # often not the set's colour
        else:
            inst = instruments[fid]
            mesh, cls, colour = scene_mesh(inst, str(rng.choice(inst.rests))), inst.coarse, None
        t = mesh.extents[2]
        x, y = rng.uniform(cx - hx, cx + hx), rng.uniform(cy - hy, cy + hy)
        T = pose(x, y, 0.0, rng.uniform(-np.pi, np.pi), flip=bool(rng.random() < 0.5))
        fp = _footprint(mesh, T, shape, origin)
        z0 = float(height[fp].max()) if fp.any() else 0.0
        T[2, 3] = z0 + t / 2
        height[fp] = z0 + t
        pile.append((mesh, T, cls, colour))
    return pile


def _real_background(rng: np.random.Generator, cam: CamSpec) -> np.ndarray | None:
    files = sorted(BG_DIR.glob("*.jpg")) if BG_DIR.is_dir() else []
    if not files:
        return None
    img = cv2.imread(str(files[rng.integers(len(files))]))
    h, w = img.shape[:2]
    s = rng.uniform(0.5, 1.0)                                   # a random crop, flipped / dimmed / tinted
    ch, cw = int(h * s), int(w * s)
    y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
    img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (cam.w, cam.h))
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1]
    return np.clip(img * rng.uniform(0.7, 1.2) + rng.normal(0, 8, 3), 0, 255).astype(np.float32)


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
        if poses_source:
            pile = [(scene_mesh(insts[fid]), T, insts[fid].coarse, None) for fid, T in poses_source(rng)]
        else:
            pile = random_pile(insts, rng, int(rng.integers(2, 9)), n_other=int(rng.integers(0, 5)))
        # scan pose (~40 cm) down to the 10 cm hover (camera ~16 cm up); fx spans the real undistorted camera
        cam = CamSpec(height_mm=rng.uniform(160.0, 450.0), fx=rng.uniform(520.0, 720.0))
        bg = _real_background(rng, cam) if rng.random() < 0.5 else None
        bgr, masks = render_scene([(m, T) for m, T, _, _ in pile], cam, rng, bg=bg, colours=[c for *_, c in pile])
        if rng.random() < 0.7:
            bgr, gm = add_gripper(bgr, rng)
            masks = [m & ~gm for m in masks]
        if rng.random() < 0.5:
            bgr = cv2.GaussianBlur(bgr, (0, 0), rng.uniform(0.3, 1.2))
        bgr = np.clip(bgr + rng.normal(0, rng.uniform(1, 5), bgr.shape), 0, 255).astype(np.uint8)
        split = "val" if i < n_val else "train"
        stem = f"{prefix}{i:05d}"
        write_yolo_sample(out / "images" / split / f"{stem}.jpg", out / "labels" / split / f"{stem}.txt",
                          bgr, masks, [names.index(cls) for _, _, cls, _ in pile])
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
