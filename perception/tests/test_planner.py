import numpy as np
import pytest

from scrub_vision.identify import identify
from scrub_vision.planner import plan
from scrub_vision.render import CamSpec, pose, render_scene, scene_mesh
from scrub_vision.templates import build_templates, load_instruments

CAM = CamSpec()
S = CAM.height_mm / CAM.fx
H = np.array([[S, 0, -CAM.w / 2 * S], [0, S, -CAM.h / 2 * S], [0, 0, 1]])


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
    p = plan(dets, "forceps_resano", H)
    assert p["action"] == "grasp_target" and p["target"]["id"] == "forceps_resano"
    assert p["place"] is None


def test_target_buried_is_not_dug_out(insts, templates):
    """Nothing is moved out of the way (user decision): a covered target is a 'can't get to it', not a move."""
    dets = scene(insts, templates, [("forceps_resano", 0, -20, 0.0, 0), ("needle_holder_ryder", 40, -20, 1.4, 1)])
    p = plan(dets, "forceps_resano", H)
    assert p["action"] == "fail" and "can't get to" in p["say"]


def test_nothing_graspable_fails(insts, templates):
    p = plan([], "forceps_resano", H)
    assert p["action"] == "fail" and "resano" in p["say"].lower()


def test_any_scalpel_satisfies_any_scalpel_request(insts, templates):
    dets = scene(insts, templates, [("scalpel_20", 0, -20, 0.5, 0)])
    p = plan(dets, "scalpel_15", H)
    assert p["action"] == "grasp_target"
    assert p["target"]["ambiguous"] is False
    assert "I think" not in p["say"]


def test_any_scalpel_is_grasped_when_a_scalpel_is_wanted(insts, templates):
    dets = scene(insts, templates, [("scalpel_10", 0, -20, 0.5, 0)])
    p = plan(dets, "scalpel_11", H)
    assert p["action"] == "grasp_target"


def test_blob_matching_no_instrument_is_never_moved(insts, templates):
    from scrub_vision.identify import Detection
    junk = Detection(mask=np.zeros((CAM.h, CAM.w), bool), coarse=None, fine_id=None, exposure=1.0, confidence=1.0)
    junk.mask[100:105, 100:400] = True            # a sheet wrinkle
    p = plan([junk], "forceps_resano", H)
    assert p["action"] == "fail"


def _det(fid, coarse, cands, exposure, x0, y0=100, conf=0.9):
    from scrub_vision.identify import Detection
    m = np.zeros((CAM.h, CAM.w), bool)
    m[y0:y0 + 10, x0:x0 + 200] = True
    gx, gy = (x0 + 100) * S - CAM.w / 2 * S, (y0 + 5) * S - CAM.h / 2 * S
    return Detection(mask=m, coarse=coarse, fine_id=fid, candidates=cands, exposure=exposure, confidence=conf,
                     axis_deg=0.0, yaw_deg=90.0, grasp_mm=(gx, gy), grasp_px=(x0 + 100, y0 + 5))


def test_target_needs_most_of_it_in_view_to_grasp():
    partial = _det("needle_holder_castroviejo", "castroviejo", ["needle_holder_castroviejo"], 0.5, 50)
    p = plan([partial], "needle_holder_castroviejo", H)
    assert p["action"] != "grasp_target"


def test_target_at_87_percent_exposure_is_grasped():
    # the real lone scalpel scored 0.87 and was refused under the old 0.9 threshold
    lone = _det("scalpel_11", "scalpel", ["scalpel_10", "scalpel_11", "scalpel_15", "scalpel_20"], 0.87, 50)
    lone.alt_ids = ["scalpel_10", "scalpel_15", "scalpel_20"]          # as identify() sets for the scalpel group
    p = plan([lone], "scalpel_10", H)
    assert p["action"] == "grasp_target"


def test_unrecognised_blob_on_top_of_target_blocks_the_grasp():
    from scrub_vision.identify import Detection
    tgt = _det("forceps_resano", "forceps", ["forceps_resano"], 0.95, 100)
    blob = Detection(mask=np.zeros((CAM.h, CAM.w), bool), coarse=None, fine_id=None, exposure=0.3, confidence=0.3)
    blob.mask[60:150, 190:215] = True          # crosses the target and its grasp point, matches no template
    tgt.mask[100:110, 190:215] = False         # the target is cut where the blob lies on it
    p = plan([tgt, blob], "forceps_resano", H)
    assert p["action"] != "grasp_target"


def test_unknown_object_on_target_blocks_it(insts, templates):
    """Something no template explains (class `other`, e.g. a random object) lying across the requested instrument:
    it is not moved; the plan says the target can't be reached."""
    import trimesh
    target = scene_mesh(insts["scalpel_10"])
    rod = trimesh.creation.box((90.0, 15.0, 6.0))                   # matches no template (all are 125+ mm)
    _, masks = render_scene([(target, pose(0, -20, target.extents[2] / 2, 0.0)), (rod, pose(0, -20, 12.0, np.pi / 2))],
                            CAM, np.random.default_rng(0))
    dets = [identify(masks[0], "scalpel", H, templates, insts), identify(masks[1], "other", H, templates, insts)]
    p = plan(dets, "scalpel_10", H)
    assert p["action"] == "fail" and "can't get to" in p["say"]


def test_forceps_on_its_side_reports_its_lying_height(insts, templates):
    m = scene_mesh(insts["forceps_fine_vascular"], "side")
    _, masks = render_scene([(m, pose(0, -20, m.extents[2] / 2, 0.3))], CAM, np.random.default_rng(0))
    d = identify(masks[0], "forceps", H, templates, insts)
    assert d.fine_id == "forceps_fine_vascular" and abs(d.thickness_mm - 12.8) < 0.5     # its width, not 8 mm
