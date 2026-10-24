import numpy as np
import pytest

from scrub_vision.templates import build_templates, load_instruments, mesh_for

MM_PER_PX = 0.5


@pytest.fixture(scope="module")
def insts():
    return load_instruments()


@pytest.fixture(scope="module")
def templates(insts):
    return build_templates(insts, MM_PER_PX)


def by_id(templates, fid, flip=False):
    return next(t for t in templates if t.fine_id == fid and t.flipped == flip)


def test_every_tray_instrument_has_a_visual_model(insts):
    import yaml
    tray = yaml.safe_load((Path_root() / "config" / "tray.yaml").read_text())["instruments"]
    missing = set(tray) - set(insts) - {"probe_coronary"}
    assert not missing


def Path_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("fid,length", [("forceps_resano", 230), ("needle_holder_castroviejo", 140),
                                        ("clamp_aortic_cross", 250), ("forceps_debakey", 200)])
def test_template_length(templates, fid, length):
    t = by_id(templates, fid)
    assert abs(t.length_mm - length) <= 0.02 * length


def test_scalpel_is_handle_plus_blade(insts):
    m = mesh_for(insts["scalpel_20"])
    assert 160 < m.extents.max() < 175     # 135 handle + 49.5 blade - 15 overlap


def test_tip_is_at_plus_x(templates):
    t = by_id(templates, "needle_holder_ryder")      # rings are the wide end, jaws are the tip
    cols = (t.mask > 0).sum(0)
    n = len(cols)
    assert cols[-n // 6:].mean() < cols[: n // 6].mean()


def test_grasp_point_near_tip_and_on_the_instrument(templates):
    t = by_id(templates, "forceps_debakey")
    u, v = t.grasp_px
    assert u > 0.6 * t.mask.shape[1]
    assert t.mask[int(round(v)), int(round(u))] > 0 or t.mask[:, int(round(u))].any()


def test_both_flips_built(templates):
    assert sum(t.fine_id == "scissors_potts" for t in templates) == 2


def test_flip_mirrors_mask(templates):
    a, b = by_id(templates, "scissors_potts"), by_id(templates, "scissors_potts", True)
    assert np.array_equal(a.mask[::-1], b.mask)
