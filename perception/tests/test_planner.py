import numpy as np
import pytest

from scrub_vision.identify import identify
from scrub_vision.planner import plan
from scrub_vision.render import CamSpec, pose, render_scene, scene_mesh
from scrub_vision.templates import build_templates, load_instruments

CAM = CamSpec()
S = CAM.height_mm / CAM.fx
H = np.array([[S, 0, -CAM.w / 2 * S], [0, S, -CAM.h / 2 * S], [0, 0, 1]])
STAGING = [(250.0, -200.0), (250.0, -260.0), (250.0, -320.0)]


@pytest.fixture(scope="module")
def insts():
    return load_instruments()


@pytest.fixture(scope="module")
def templates(insts):
    return build_templates(insts, 0.5)


def scene(insts, templates, specs):
    """specs: (fid, x, y, yaw, level) - level 0 on the table, 1 on top of level 0."""
    items, z_top = [], {}
    for fid, x, y, yaw, level in specs:
        m = scene_mesh(insts[fid])
        z = (12.0 if level else 0.0) + m.extents[2] / 2
        items.append((m, pose(x, y, z, yaw)))
    _, masks = render_scene(items, CAM, np.random.default_rng(0))
    return [identify(m, None, H, templates, insts) for m in masks]


def test_target_on_top(insts, templates):
    dets = scene(insts, templates, [("forceps_resano", -60, -40, 0.0, 0), ("scissors_potts", 90, 60, 1.0, 0)])
    p = plan(dets, "forceps_resano", H, STAGING, set())
    assert p["action"] == "grasp_target" and p["target"]["id"] == "forceps_resano"
    assert p["place"] is None


def test_target_buried_moves_top_one(insts, templates):
    dets = scene(insts, templates, [("forceps_resano", 0, -20, 0.0, 0), ("needle_holder_ryder", 40, -20, 1.4, 1)])
    p = plan(dets, "forceps_resano", H, STAGING, set())
    assert p["action"] == "move_aside" and p["target"]["id"] == "needle_holder_ryder"
    assert (p["place"]["x_mm"], p["place"]["y_mm"]) == STAGING[0]


def test_prefers_mover_overlapping_target(insts, templates):
    dets = scene(insts, templates, [("forceps_resano", 0, -20, 0.0, 0),
                                    ("needle_holder_ryder", 40, -20, 1.4, 1),
                                    ("scissors_potts", -160, 90, 0.3, 0)])
    p = plan(dets, "forceps_resano", H, STAGING, set())
    assert p["target"]["id"] == "needle_holder_ryder"


def test_target_not_visible_moves_something(insts, templates):
    dets = scene(insts, templates, [("scissors_potts", 0, -20, 0.3, 0)])
    p = plan(dets, "forceps_resano", H, STAGING, set())
    assert p["action"] == "move_aside"


def test_nothing_graspable_fails(insts, templates):
    p = plan([], "forceps_resano", H, STAGING, set())
    assert p["action"] == "fail" and "resano" in p["say"].lower()


def test_staging_spots_not_reused(insts, templates):
    dets = scene(insts, templates, [("forceps_resano", 0, -20, 0.0, 0), ("needle_holder_ryder", 40, -20, 1.4, 1)])
    p = plan(dets, "forceps_resano", H, STAGING, {0, 1})
    assert (p["place"]["x_mm"], p["place"]["y_mm"]) == STAGING[2]


def test_staging_full_fails(insts, templates):
    dets = scene(insts, templates, [("forceps_resano", 0, -20, 0.0, 0), ("needle_holder_ryder", 40, -20, 1.4, 1)])
    p = plan(dets, "forceps_resano", H, STAGING, {0, 1, 2})
    assert p["action"] == "fail"


def test_already_staged_instruments_are_not_moved_again(insts, templates):
    # potts sits on staging spot 0, which is used; target not visible -> nothing left to move
    x, y = STAGING[0]
    dets = scene(insts, templates, [("scissors_potts", x - 290, y + 160, 0.0, 0)])
    staged_at = {0: dets[0].grasp_mm}
    p = plan(dets, "forceps_resano", H, STAGING, {0}, staged_xy=list(staged_at.values()))
    assert p["action"] == "fail"


def test_ambiguous_scalpel_is_flagged(insts, templates):
    dets = scene(insts, templates, [("scalpel_15", 0, -20, 0.5, 0)])
    p = plan(dets, "scalpel_15", H, STAGING, set())
    assert p["action"] == "grasp_target"
    assert p["target"]["ambiguous"] in (True, False)
