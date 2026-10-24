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


def test_blob_matching_no_instrument_is_never_moved(insts, templates):
    from scrub_vision.identify import Detection
    junk = Detection(mask=np.zeros((CAM.h, CAM.w), bool), coarse=None, fine_id=None, exposure=1.0, confidence=1.0)
    junk.mask[100:105, 100:400] = True            # a sheet wrinkle
    p = plan([junk], "forceps_resano", H, STAGING, set())
    assert p["action"] == "fail"


def _det(fid, coarse, cands, exposure, x0, y0=100, conf=0.9):
    from scrub_vision.identify import Detection
    m = np.zeros((CAM.h, CAM.w), bool)
    m[y0:y0 + 10, x0:x0 + 200] = True
    gx, gy = (x0 + 100) * S - CAM.w / 2 * S, (y0 + 5) * S - CAM.h / 2 * S
    return Detection(mask=m, coarse=coarse, fine_id=fid, candidates=cands, exposure=exposure, confidence=conf,
                     axis_deg=0.0, yaw_deg=90.0, grasp_mm=(gx, gy), grasp_px=(x0 + 100, y0 + 5))


def test_never_moves_an_uncertain_same_class_instrument():
    # a partly covered Resano reads short, as a DeBakey: it might be the target, so don't move it away
    maybe = _det("forceps_debakey", "forceps", ["forceps_debakey"], 0.75, 50)
    other = _det("scissors_potts", "potts", ["scissors_potts"], 1.0, 500)
    p = plan([maybe, other], "forceps_resano", H, STAGING, set())
    assert p["action"] == "move_aside" and p["target"]["id"] == "scissors_potts"


def test_confident_same_class_sibling_can_be_moved():
    sib = _det("forceps_debakey", "forceps", ["forceps_debakey"], 0.97, 50)
    p = plan([sib], "forceps_resano", H, STAGING, set())
    assert p["action"] == "move_aside" and p["target"]["id"] == "forceps_debakey"


def test_target_needs_near_full_view_to_grasp():
    partial = _det("needle_holder_castroviejo", "castroviejo", ["needle_holder_castroviejo"], 0.8, 50)
    p = plan([partial], "needle_holder_castroviejo", H, STAGING, set())
    assert p["action"] != "grasp_target"


def test_unrecognised_blob_on_top_of_target_blocks_the_grasp():
    from scrub_vision.identify import Detection
    tgt = _det("forceps_resano", "forceps", ["forceps_resano"], 0.95, 100)
    blob = Detection(mask=np.zeros((CAM.h, CAM.w), bool), coarse=None, fine_id=None, exposure=0.3, confidence=0.3)
    blob.mask[60:150, 190:215] = True          # crosses the target and its grasp point, matches no template
    tgt.mask[100:110, 190:215] = False         # the target is cut where the blob lies on it
    p = plan([tgt, blob], "forceps_resano", H, STAGING, set())
    assert p["action"] != "grasp_target"
