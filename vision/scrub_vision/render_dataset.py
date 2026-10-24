"""Render a synthetic YOLO dataset from the STL files: one class per instrument id in config/tray.yaml.

    python -m scrub_vision.render_dataset --train 4000 --val 600
    python -m scrub_vision.render_dataset --preview 12      # contact sheet only, to eyeball the renders

Pure numpy + OpenCV (top-down perspective camera, flat shading, painter's algorithm): no GL needed.
Class index = position in tray.yaml `instruments:`, so the model's labels are the shared instrument ids.
"""
from __future__ import annotations

import argparse
import fnmatch
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml
from scrub_arm.config import load_tray_config

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "assets" / "stl"
OUT = ROOT / "data" / "synth"
SIZE = 640
TURQUOISE = np.array([64, 224, 208.0])  # RGB; shade and white balance still vary per image

# instrument id -> STL glob(s) (one random variant per rendered instance). Scalpels are blade + handle.
MESHES = {
    "needle_holder_castroviejo": ["Castroviejo*"],
    "needle_holder_ryder": ["Ryder*"],
    "needle_holder_long": ["Mayo-Hegar*long*"],
    "needle_holder_short": ["Mayo-Hegar*short*"],
    "forceps_debakey": ["DeBakey*"],
    "forceps_resano": ["Resano*"],
    "forceps_fine_vascular": ["Fine vascular*"],
    "scissors_potts": ["Potts*"],
    "scissors_micro": ["Micro*forward*", "Micro*back*"],
    "clamp_aortic_cross": ["Aortic*"],
    "clamp_side_biting": ["Side-biting*"],
}
BLADES = {"scalpel_10": "Scalpel blade #10*", "scalpel_11": "Scalpel blade #11*",
          "scalpel_15": "Scalpel blade #15*", "scalpel_20": "Scalpel blade #20*"}
HANDLES = {"scalpel_10": ["Scalpel handle #3 *", "Scalpel handle #3L*", "Scalpel handle #7*"],
           "scalpel_20": ["Scalpel handle #4*"]}
HANDLES |= {k: HANDLES["scalpel_10"] for k in ("scalpel_11", "scalpel_15")}
BLADE_INSERT_MM = 12      # how far the blade tang sits inside the handle
BLADE_AT_MAX_END = True   # which handle end carries the blade (checked on the --preview sheet)


def _load(pattern: str) -> trimesh.Trimesh:
    hits = [p for p in sorted(RAW.rglob("*.stl")) if fnmatch.fnmatch(p.name, pattern)]
    if not hits:
        raise FileNotFoundError(f"no STL matches {pattern!r} under {RAW}")
    return trimesh.load(hits[0])


def _probe() -> trimesh.Trimesh:
    # shortcut: no STL exists for the coronary probe, so this is a stand-in rod with a ball tip and grip; replace when we have the real part.
    rod = trimesh.creation.cylinder(radius=1.5, height=150)
    rod.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    tip = trimesh.creation.icosphere(radius=2.6)
    tip.apply_translation([75, 0, 0])
    grip = trimesh.creation.cylinder(radius=4, height=60)
    grip.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    return trimesh.util.concatenate([rod, tip, grip])


def _scalpel(iid: str, rng: np.random.Generator) -> trimesh.Trimesh:
    blade, handle = _load(BLADES[iid]), _load(HANDLES[iid][rng.integers(len(HANDLES[iid]))])
    h, b = handle.vertices - handle.bounds.mean(0), blade.vertices - blade.bounds.mean(0)
    hx = np.ptp(h[:, 0])
    b = b - [b[:, 0].min(), 0, 0]  # tang at x=0, tip at +x
    if BLADE_AT_MAX_END:
        b[:, 0] += hx / 2 - BLADE_INSERT_MM
    else:
        b[:, 0] = -b[:, 0] - hx / 2 + BLADE_INSERT_MM
        b[:, 1] = -b[:, 1]
    return trimesh.Trimesh(np.vstack([h, b]), np.vstack([handle.faces, blade.faces + len(h)]), process=False)


def _align(m: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Centre, then PCA-rotate so x = long axis, y = width, z = thinnest axis (the way it lies on a table)."""
    v = m.vertices - m.vertices.mean(0)
    _, vec = np.linalg.eigh(np.cov(v.T))
    vec = vec[:, ::-1]
    if np.linalg.det(vec) < 0:
        vec[:, 2] *= -1
    return v @ vec, m.faces


def build_pool(ids: list[str]) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Fails loudly if a tray.yaml instrument has no mesh recipe, so classes can't silently drift."""
    unknown = [i for i in ids if i not in MESHES and i not in BLADES and i != "probe_coronary"]
    if unknown:
        raise KeyError(f"tray.yaml instruments with no mesh recipe in render_dataset.py: {unknown}")
    rng = np.random.default_rng(0)
    pool = {}
    for iid in ids:
        if iid in BLADES:
            variants = [_align(_scalpel(iid, rng)) for _ in range(8)]
        elif iid == "probe_coronary":
            variants = [_align(_probe())]
        else:
            variants = [_align(_load(p)) for p in MESHES[iid]]
        pool[iid] = variants
    return pool


def _rot(axis: int, a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    r = np.eye(3)
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def _render_tool(v, f, color, light, scale, H, center):
    """Return (rgb layer, mask) of one tool seen from a camera H mm above the table."""
    depth = H - v[:, 2]
    px = center + scale * H * v[:, :2] / depth[:, None]
    tri, tri3 = px[f], v[f]
    n = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    keep = n[:, 2] > 0  # facing the camera
    lam = np.clip(n @ light, 0, None)
    hv = light + [0, 0, 1]
    spec = np.clip(n @ (hv / np.linalg.norm(hv)), 0, None) ** 24
    # Lambert + darker near the table: with one colour, shading is the only depth cue the model gets.
    shade = (0.35 + 0.65 * lam) * (0.8 + 0.2 * tri3[:, :, 2].mean(1) / (v[:, 2].max() + 1e-9))
    layer = np.zeros((SIZE, SIZE, 3), np.uint8)
    mask = np.zeros((SIZE, SIZE), np.uint8)
    for k in np.argsort(tri3[:, :, 2].mean(1)):  # low first, high on top
        if not keep[k]:
            continue
        pts = np.round(tri[k] * 16).astype(np.int32)
        col = np.clip(color * shade[k] + 255 * 0.25 * spec[k], 0, 255)
        cv2.fillConvexPoly(layer, pts, tuple(float(c) for c in col), cv2.LINE_AA, 4)
        cv2.fillConvexPoly(mask, pts, 255, cv2.LINE_AA, 4)
    return layer, mask


def _background(rng) -> np.ndarray:
    base = rng.choice([[200, 200, 195], [150, 150, 150], [90, 150, 90], [150, 120, 70], [235, 235, 235],
                       [60, 60, 70]]) + rng.normal(0, 12, 3)
    # mostly brightness blotches (cloth/tray wear), not rainbow chroma noise
    low = cv2.resize(rng.normal(0, 14, (6, 6, 1)).astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)[..., None]
    img = np.clip(base + low + rng.normal(0, 4, (SIZE, SIZE, 3)), 0, 255).astype(np.float32)
    gx = np.linspace(rng.uniform(0.8, 1.1), rng.uniform(0.8, 1.1), SIZE)[None, :, None]
    return img * gx


def render_scene(pool, ids, rng, scale_range=(0.9, 1.6)):
    """One image + YOLO label rows. Tools are placed without overlapping so every label box is clean."""
    img, occ = _background(rng), np.zeros((SIZE, SIZE), np.uint8)
    H = rng.uniform(380, 600)
    scale = rng.uniform(*scale_range) / (H / 450) # px per mm at table level, roughly constant across H
    light = rng.normal([0, 0, 1.2], [0.5, 0.5, 0.2]); light /= np.linalg.norm(light)
    n = int(rng.integers(3, 9))
    picks = list(rng.permutation(ids)[:n])
    if rng.random() < 0.5:  # hard case: two scalpels side by side
        picks[:2] = list(rng.choice([i for i in ids if i.startswith("scalpel")], 2, replace=False))
    labels = []
    for iid in picks:
        v0, f = pool[iid][rng.integers(len(pool[iid]))]
        roll = rng.choice([0, np.pi, np.pi / 2, 3 * np.pi / 2], p=[0.4, 0.4, 0.1, 0.1]) + rng.normal(0, 0.1)
        v = v0 @ (_rot(1, rng.normal(0, 0.05)) @ _rot(0, roll)).T
        v[:, 2] -= v[:, 2].min()
        v = v @ _rot(2, rng.uniform(0, 2 * np.pi)).T
        color = np.clip(TURQUOISE + rng.normal(0, 10, 3), 0, 255)  # every printed tool is the same turquoise
        for _ in range(25):
            pos = rng.uniform(SIZE * 0.1, SIZE * 0.9, 2)
            layer, mask = _render_tool(v, f, color[::-1], light, scale, H, pos)
            ys, xs = np.nonzero(mask > 127)
            if len(xs) == 0 or xs.min() < 2 or ys.min() < 2 or xs.max() > SIZE - 3 or ys.max() > SIZE - 3:
                continue
            if (cv2.bitwise_and(mask, occ) > 0).any():
                continue
            a = (mask[..., None] / 255.0)
            shadow = cv2.GaussianBlur(np.roll(mask, (int(4 * light[1] * -3) + 3, int(4 * light[0] * -3) + 3), (0, 1)),
                                      (0, 0), rng.uniform(2, 5))[..., None] / 255.0
            img = img * (1 - 0.35 * shadow) * (1 - a) + layer * a
            occ = cv2.bitwise_or(occ, cv2.dilate(mask, np.ones((9, 9), np.uint8)))
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            labels.append((ids.index(iid), (x0 + x1) / 2 / SIZE, (y0 + y1) / 2 / SIZE, (x1 - x0) / SIZE, (y1 - y0) / SIZE))
            break
    img = np.clip(img * rng.uniform(0.8, 1.15) + rng.uniform(-15, 15), 0, 255)
    if rng.random() < 0.6:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 1.8))
    img = np.clip(img + rng.normal(0, rng.uniform(1, 6), img.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), labels


_POOL = _IDS = None


def _init(ids):
    global _POOL, _IDS
    _IDS, _POOL = ids, build_pool(ids)


def _job(args):
    split, i, seed = args
    img, labels = render_scene(_POOL, _IDS, np.random.default_rng(seed))
    stem = f"{split}_{i:05d}"
    cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), img)
    (OUT / "labels" / split / f"{stem}.txt").write_text(
        "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in labels))
    return len(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=600)
    ap.add_argument("--preview", type=int, help="write a contact sheet of N scenes to data/preview.jpg and exit")
    args = ap.parse_args()

    ids = list(load_tray_config().instruments)
    if args.preview:
        pool, rng = build_pool(ids), np.random.default_rng(1)
        tiles = [cv2.resize(render_scene(pool, ids, rng)[0], (426, 426)) for _ in range(args.preview)]
        cols = 4
        rows = [np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles) - cols + 1, cols)]
        (ROOT / "data").mkdir(exist_ok=True)
        cv2.imwrite(str(ROOT / "data" / "preview.jpg"), np.vstack(rows))
        return
    for sub in ("images", "labels"):
        for split in ("train", "val"):
            (OUT / sub / split).mkdir(parents=True, exist_ok=True)
    jobs = [("train", i, 1000 + i) for i in range(args.train)] + [("val", i, 10_000_000 + i) for i in range(args.val)]
    with mp.Pool(initializer=_init, initargs=(ids,)) as pool:
        counts = []
        for i, n in enumerate(pool.imap_unordered(_job, jobs, chunksize=20), 1):
            counts.append(n)
            if i % 500 == 0:
                print(f"rendered {i}/{len(jobs)}", flush=True)
    (OUT / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(OUT), "train": "images/train", "val": "images/val", "names": dict(enumerate(ids))}))
    print(f"rendered {len(jobs)} images, {sum(counts)} labeled tools -> {OUT}")


if __name__ == "__main__":
    main()
