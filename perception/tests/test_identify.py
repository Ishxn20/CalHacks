import numpy as np
import pytest

from scrub_vision.identify import feasible, identify, measure, segment_hsv
from scrub_vision.render import CamSpec, pose, render_scene, scene_mesh
from scrub_vision.templates import build_templates, load_instruments

CAM = CamSpec()


def table_H(cam=CAM):
    """px -> mm on the table plane for the synthetic camera (y kept as image-down)."""
    s = cam.height_mm / cam.fx
    return np.array([[s, 0, -cam.w / 2 * s], [0, s, -cam.h / 2 * s], [0, 0, 1]])


@pytest.fixture(scope="module")
def insts():
    return load_instruments()


@pytest.fixture(scope="module")
def templates(insts):
    return build_templates(insts, 0.5)


def render(insts, specs):
    items = [(scene_mesh(insts[f]), pose(x, y, z, yaw, flip))
             for f, x, y, z, yaw, flip in specs]
    return render_scene(items, CAM, np.random.default_rng(0))


def one(insts, fid, x=0, y=-20, yaw=0.4, flip=False):
    t = scene_mesh(insts[fid]).extents[2]
    return render(insts, [(fid, x, y, t / 2, yaw, flip)])[1][0]


@pytest.mark.parametrize("fid", ["forceps_resano", "forceps_debakey", "forceps_fine_vascular",
                                 "needle_holder_ryder", "needle_holder_long", "clamp_aortic_cross",
                                 "scissors_potts", "scalpel_20"])
def test_identify_single_rendered(insts, templates, fid):
    mask = one(insts, fid, yaw=0.4)
    d = identify(mask, None, table_H(), templates, insts)
    assert d.fine_id == fid
    axis = np.degrees(0.4)
    assert min(abs((d.axis_deg - axis + 90) % 180 - 90), 180) < 5
    assert abs(((d.yaw_deg - d.axis_deg) % 180) - 90) < 1e-6      # jaws open across the shank


def test_coarse_class_restricts_candidates(insts, templates):
    mask = one(insts, "needle_holder_ryder")
    d = identify(mask, "forceps", table_H(), templates, insts)
    assert d.fine_id is None


def test_identify_flipped_grasp_at_tip_end(insts, templates):
    fid, yaw = "scissors_potts", 1.0
    mask = one(insts, fid, yaw=yaw, flip=True)
    d = identify(mask, None, table_H(), templates, insts)
    assert d.fine_id == fid
    # tip of the instrument is +x in its own frame -> world direction (cos yaw, sin yaw) from centre (0, -20)
    v = np.array(d.grasp_mm) - [0, -20]
    assert np.dot(v, [np.cos(yaw), np.sin(yaw)]) > 20


def test_length_gate_rejects_merged_blob(insts, templates):
    t = scene_mesh(insts["forceps_debakey"]).extents[2]
    _, masks = render(insts, [("forceps_debakey", -95, -20, t / 2, 0.0, False),
                              ("forceps_resano", 110, -20, t / 2, 0.0, False)])
    merged = masks[0] | masks[1]
    d = identify(merged, None, table_H(), templates, insts)
    assert d.fine_id is None


def test_exposure_drops_when_occluded(insts, templates):
    ta = scene_mesh(insts["forceps_resano"]).extents[2]
    tb = scene_mesh(insts["clamp_side_biting"]).extents[2]
    _, (under, top) = render(insts, [("forceps_resano", 0, -20, ta / 2, 0.0, False),
                                     ("clamp_side_biting", 20, -15, ta + tb / 2, 0.15, False)])
    from scrub_vision.identify import on_top
    du = identify(under, None, table_H(), templates, insts)
    dt = identify(top, None, table_H(), templates, insts)
    assert dt.exposure > du.exposure
    assert on_top(dt, [du]) and not on_top(du, [dt])
    assert du.fine_id == "forceps_resano"          # still identified while partly covered


def test_feasible_rejects_jaw_over_upper_mask(insts, templates):
    ta = scene_mesh(insts["forceps_resano"]).extents[2]
    _, (under, top) = render(insts, [("forceps_resano", 0, -20, ta / 2, 0.0, False),
                                     ("forceps_debakey", 46, -20, ta + 5, np.pi / 2, False)])
    H = table_H()
    du = identify(under, None, H, templates, insts)
    dt = identify(top, None, H, templates, insts)
    # Resano's grasp is ~0.3 of its length from the tip (+x end, ~46 mm from centre): DeBakey lies across it
    assert not feasible(du, [dt], H)
    assert feasible(dt, [du], H)


def test_measure_length(insts):
    mask = one(insts, "forceps_resano", yaw=0.0)
    length, width, axis, _ = measure(mask, table_H())
    assert abs(length - 230) < 0.05 * 230


def test_segment_hsv_finds_green_blobs_outside_gripper(insts):
    t = scene_mesh(insts["forceps_resano"]).extents[2]
    bgr, _ = render(insts, [("forceps_resano", -80, -60, t / 2, 0.0, False),
                            ("scissors_potts", 90, 40, t / 2, 1.0, False)])
    gm = np.zeros(bgr.shape[:2], bool)
    gm[int(bgr.shape[0] * 0.75):] = True
    blobs = segment_hsv(bgr, gripper_mask=gm)
    assert len(blobs) == 2


def test_above_uses_the_crossing(insts, templates):
    from scrub_vision.identify import above
    ta = scene_mesh(insts["forceps_resano"]).extents[2]
    _, (under, top) = render(insts, [("forceps_resano", 0, -20, ta / 2, 0.0, False),
                                     ("forceps_debakey", 0, -20, ta + 5, np.pi / 2, False)])
    assert above(top, under) and not above(under, top)


def test_separate_instruments_are_not_above_each_other(insts):
    from scrub_vision.identify import above
    t = scene_mesh(insts["forceps_resano"]).extents[2]
    _, (a, b) = render(insts, [("forceps_resano", -80, -60, t / 2, 0.0, False),
                               ("scissors_potts", 90, 60, t / 2, 1.0, False)])
    assert not above(a, b) and not above(b, a)


def test_scalpel_blades_are_candidates_not_a_guess(insts, templates):
    d = identify(one(insts, "scalpel_15"), None, table_H(), templates, insts)
    assert d.fine_id in (None, "scalpel_15")
    assert "scalpel_15" in d.candidates


def test_castroviejo_vs_micro_scissors_by_length(insts, templates):
    d = identify(one(insts, "needle_holder_castroviejo"), None, table_H(), templates, insts)
    assert d.fine_id == "needle_holder_castroviejo"
