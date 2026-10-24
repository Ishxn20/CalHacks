"""Instrument table + top-down silhouette templates rendered from the STLs.

Convention for every template: instrument lying flat (resting on its largest face), long axis along
+x, working end (tip) at +x. `grasp_px` is where the jaws should close, in template pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yaml

PKG = Path(__file__).resolve().parents[1]
STL_DIR = PKG / "assets" / "stl"
INSTRUMENTS_YAML = PKG / "config" / "instruments.yaml"


@dataclass
class Instrument:
    id: str
    coarse: str
    stl: list[str]
    tip: str
    grasp_frac: float
    thickness_mm: float
    same_look_as: str | None = None


@dataclass
class Template:
    fine_id: str
    coarse: str
    mask: np.ndarray          # uint8, 255 inside
    length_mm: float
    grasp_px: tuple[float, float]
    flipped: bool
    mm_per_px: float
    _canon: object = field(default=None, repr=False, compare=False)   # identify.py's cached polygons


_cfg_cache: dict | None = None


def _cfg() -> dict:
    global _cfg_cache
    if _cfg_cache is None:
        _cfg_cache = yaml.safe_load(INSTRUMENTS_YAML.read_text())
    return _cfg_cache


def classes() -> list[str]:
    return list(_cfg()["classes"])


def load_instruments(path: Path | None = None) -> dict[str, Instrument]:
    d = yaml.safe_load(path.read_text()) if path else _cfg()
    out = {}
    for iid, v in d["instruments"].items():
        stl = v["stl"] if isinstance(v["stl"], list) else [v["stl"]]
        out[iid] = Instrument(id=iid, coarse=v["coarse"], stl=stl, tip=v["tip"], grasp_frac=float(v["grasp_frac"]),
                              thickness_mm=float(v["thickness_mm"]), same_look_as=v.get("same_look_as"))
    return out


def _flat(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate so the long axis is x and the thinnest extent is z (lying on its largest face), min corner at 0."""
    m = mesh.copy()
    e = m.extents
    thin = int(np.argmin(e))
    other = [i for i in range(3) if i != thin]
    long_ = max(other, key=lambda i: e[i])
    mid = [i for i in other if i != long_][0]
    m.vertices = m.vertices[:, [long_, mid, thin]]
    m.vertices -= m.vertices.min(0)
    return m


_mesh_cache: dict[str, trimesh.Trimesh] = {}


def mesh_for(inst: Instrument) -> trimesh.Trimesh:
    """Flat mesh in mm with the tip at +x (scalpel = handle + blade joined end to end)."""
    if inst.id in _mesh_cache:
        return _mesh_cache[inst.id]
    parts = [_flat(trimesh.load(STL_DIR / f)) for f in inst.stl]
    if len(parts) > 1:
        handle, blade = parts
        overlap = float(_cfg().get("blade_overlap_mm", 15.0))
        blade.vertices += [handle.extents[0] - overlap, (handle.extents[1] - blade.extents[1]) / 2, 0]
        m = trimesh.util.concatenate([handle, blade])
    else:
        m = parts[0]
        if inst.tip == "min_x":
            m.vertices[:, 0] = m.extents[0] - m.vertices[:, 0]
            m.faces = m.faces[:, ::-1]
    _mesh_cache[inst.id] = m
    return m


def fill_tris(img: np.ndarray, tris: np.ndarray, color) -> None:
    """Union-fill triangles. (cv2.fillPoly with many polygons uses even-odd: overlaps would cancel.)"""
    for t in np.round(tris).astype(np.int32):
        cv2.fillConvexPoly(img, t, color)


def rasterize_xy(vertices_xy: np.ndarray, faces: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    img = np.zeros(shape, np.uint8)
    fill_tris(img, vertices_xy[faces], 255)
    return img


def silhouette(mesh: trimesh.Trimesh, mm_per_px: float, flip: bool = False) -> np.ndarray:
    """Top-down mask of a flat mesh, 2 px border; `flip` mirrors it about the long axis."""
    xy = mesh.vertices[:, :2] / mm_per_px + 2
    h = int(np.ceil(mesh.extents[1] / mm_per_px)) + 4
    w = int(np.ceil(mesh.extents[0] / mm_per_px)) + 4
    m = rasterize_xy(xy, mesh.faces, (h, w))
    return m[::-1].copy() if flip else m


def _grasp_px(mask: np.ndarray, frac: float) -> tuple[float, float]:
    xs = np.where(mask.any(0))[0]
    u = xs.max() - frac * (xs.max() - xs.min())
    col = np.where(mask[:, int(round(u))] > 0)[0]
    v = float(col.mean()) if len(col) else mask.shape[0] / 2
    return float(u), v


def build_templates(instruments: dict[str, Instrument], mm_per_px: float) -> list[Template]:
    out = []
    for inst in instruments.values():
        mesh = mesh_for(inst)
        for flip in (False, True):
            mask = silhouette(mesh, mm_per_px, flip)
            out.append(Template(fine_id=inst.id, coarse=inst.coarse, mask=mask, length_mm=float(mesh.extents[0]),
                                grasp_px=_grasp_px(mask, inst.grasp_frac), flipped=flip, mm_per_px=mm_per_px))
    return out
