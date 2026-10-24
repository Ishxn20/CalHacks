import math

import numpy as np
import pytest

from scrub_arm.grasp import GrabError, point_down
from scrub_arm.vision import SimCamera, detect, learn_color, load_vision
from scrub_arm.visual_grasp import calibrate_camera, find_and_grab

ROUGHLY_DOWN = np.radians([0, 68, 46, -48, 0, 0])


@pytest.fixture
def rig(app, config_dir):
    """Arm pointing down ~7 cm over the table, a sim camera, and the object right under the jaws."""
    d, kin = app.driver, app.kin
    d.move_to(ROUGHLY_DOWN)
    point_down(d, kin)
    tip = kin.fk(d.q_command())[:3, 3]
    app.cfg.grab.table_z = float(tip[2]) - 0.07
    cam = SimCamera(d, kin, app.cfg.grab.table_z, obj_xy=tip[:2])
    vcfg = load_vision(config_dir / "vision.yaml")
    frame = cam.render()
    det_px = np.argwhere(np.all(frame == (30, 120, 230), axis=2)).mean(axis=0)  # (row, col) of the disc
    vcfg.hsv_lo, vcfg.hsv_hi, _ = learn_color(frame, int(det_px[1]), int(det_px[0]))
    return app, cam, vcfg


def test_learned_color_detects_the_object(rig):
    app, cam, vcfg = rig
    det, mask = detect(cam.render(), vcfg)
    assert det is not None and det.area > 100


def test_calibrate_then_find_moved_object(rig):
    app, cam, vcfg = rig
    d, kin = app.driver, app.kin
    msg = calibrate_camera(d, kin, cam, vcfg)
    assert "calibrated" in msg and vcfg.calibrated
    assert (vcfg.path).exists()

    cam.obj_xy = cam.obj_xy + np.array([0.025, -0.02])  # someone nudged the band
    res = find_and_grab(d, kin, cam, vcfg)
    assert res.grab.holding and res.steps >= 1
    # The jaws came down on the object (checked at the grasp height, before the lift).
    tip_xy = kin.fk(d.q_command())[:2, 3]
    assert np.linalg.norm(tip_xy - cam.obj_xy) < 0.004


def test_find_after_wrist_rotation(rig):
    """Calibration is stored in the tool frame, so a rotated wrist still servos correctly."""
    app, cam, vcfg = rig
    d, kin = app.driver, app.kin
    calibrate_camera(d, kin, cam, vcfg)
    q = d.q_command()
    q[5] += math.radians(40)
    d.move_to(q)
    cam.obj_xy = cam.obj_xy + np.array([-0.015, 0.02])
    res = find_and_grab(d, kin, cam, vcfg)
    assert res.grab.holding
    assert np.linalg.norm(kin.fk(d.q_command())[:2, 3] - cam.obj_xy) < 0.004


def test_object_out_of_view(rig):
    app, cam, vcfg = rig
    calibrate_camera(app.driver, app.kin, cam, vcfg)
    cam.obj_xy = cam.obj_xy + np.array([0.5, 0.5])
    with pytest.raises(GrabError, match="can't see"):
        find_and_grab(app.driver, app.kin, cam, vcfg)


def test_needs_calibration(rig):
    app, cam, vcfg = rig
    with pytest.raises(GrabError, match="calibrate"):
        find_and_grab(app.driver, app.kin, cam, vcfg)


def test_automatic_search_from_folded_pose(app, config_dir):
    """Real-table setup: table at z=0, arm folded at rest, band off to one side."""
    from scrub_arm.visual_grasp import _move_to_down_pose, search_and_grab

    d, kin = app.driver, app.kin
    app.cfg.grab.table_z = 0.0
    # one-time setup: hover over the band, learn its colour, calibrate
    _move_to_down_pose(d, kin, [0.26, 0.0, 0.07], 0.0)
    tip = kin.fk(d.q_command())[:3, 3]
    cam = SimCamera(d, kin, 0.0, obj_xy=tip[:2], f_px=900)  # narrower view, so the search has to sweep
    vcfg = load_vision(config_dir / "vision.yaml")
    frame = cam.render()
    rc = np.argwhere(np.all(frame == (30, 120, 230), axis=2)).mean(axis=0)
    vcfg.hsv_lo, vcfg.hsv_hi, _ = learn_color(frame, int(rc[1]), int(rc[0]))
    calibrate_camera(d, kin, cam, vcfg)

    # later: arm parked folded, band somewhere it can't see from the start position
    d.move_to(np.zeros(6))
    cam.obj_xy = np.array([0.33, -0.13])
    seen = []
    res = search_and_grab(d, kin, cam, vcfg, status=seen.append)
    assert res.grab.holding
    assert any(m.startswith("searching") for m in seen)
    assert np.linalg.norm(kin.fk(d.q_command())[:2, 3] - cam.obj_xy) < 0.004


def test_automatic_search_gives_up_cleanly(app, config_dir):
    from scrub_arm.visual_grasp import _move_to_down_pose, search_and_grab

    d, kin = app.driver, app.kin
    app.cfg.grab.table_z = 0.0
    _move_to_down_pose(d, kin, [0.26, 0.0, 0.07], 0.0)
    tip = kin.fk(d.q_command())[:3, 3]
    cam = SimCamera(d, kin, 0.0, obj_xy=tip[:2])
    vcfg = load_vision(config_dir / "vision.yaml")
    frame = cam.render()
    rc = np.argwhere(np.all(frame == (30, 120, 230), axis=2)).mean(axis=0)
    vcfg.hsv_lo, vcfg.hsv_hi, _ = learn_color(frame, int(rc[1]), int(rc[0]))
    calibrate_camera(d, kin, cam, vcfg)
    cam.obj_xy = np.array([1.0, 1.0])  # not on the table
    with pytest.raises(GrabError, match="searched the whole area"):
        search_and_grab(d, kin, cam, vcfg)


def _calibrated_rig(app, config_dir, **cam_kw):
    """Table at z=0; colour learned and pointing-down calibration done, like the one-time setup."""
    from scrub_arm.visual_grasp import _move_to_down_pose

    d, kin = app.driver, app.kin
    app.cfg.grab.table_z = 0.0
    _move_to_down_pose(d, kin, [0.26, 0.0, 0.07], 0.0)
    tip = kin.fk(d.q_command())[:3, 3]
    cam = SimCamera(d, kin, 0.0, obj_xy=tip[:2], **cam_kw)
    vcfg = load_vision(config_dir / "vision.yaml")
    frame = cam.render()
    rc = np.argwhere(np.all(frame == (30, 120, 230), axis=2)).mean(axis=0)
    vcfg.hsv_lo, vcfg.hsv_hi, _ = learn_color(frame, int(rc[1]), int(rc[0]))
    calibrate_camera(d, kin, cam, vcfg)
    return d, kin, cam, vcfg


def test_sightline_points_at_the_object_once_aimed(app, config_dir):
    from scrub_arm.visual_grasp import LOOK_ELBOW_DEG, _aim, sightline_hit

    d, kin, cam, vcfg = _calibrated_rig(app, config_dir)
    q2, q3 = LOOK_ELBOW_DEG
    d.move_to(np.radians([10, q2, q3, q2 - q3 - 55, 0, 0]))
    cam.obj_xy = np.array([0.31, 0.07])
    _aim(d, kin, cam, vcfg, 0.0)
    est = sightline_hit(kin, d.q_command(), vcfg, 0.0)
    assert np.linalg.norm(est - cam.obj_xy) < 0.01  # within 1 cm from the side view alone


def test_look_aim_approach_rotate_grab_from_folded(app, config_dir):
    from scrub_arm.visual_grasp import look_and_grab

    d, kin, cam, vcfg = _calibrated_rig(app, config_dir)
    d.move_to(np.zeros(6))              # parked folded
    cam.obj_xy = np.array([0.29, -0.11])
    seen = []
    res = look_and_grab(d, kin, cam, vcfg, status=seen.append)
    assert res.grab.holding
    assert any(m.startswith("looking") for m in seen)
    assert any("rotating to point down" in m for m in seen)
    assert np.linalg.norm(kin.fk(d.q_command())[:2, 3] - cam.obj_xy) < 0.004


def _scene(bg, obj, size=(240, 320)):
    import cv2

    img = np.full((*size, 3), bg, np.uint8)
    cv2.rectangle(img, (140, 100), (200, 130), obj, -1)
    return img


def test_coloured_click_never_matches_grey_background():
    from scrub_arm.vision import VisionCfg

    img = _scene(bg=(150, 150, 150), obj=(30, 120, 230))  # orange part on grey
    lo, hi, note = learn_color(img, 170, 115)
    vcfg = VisionCfg(path=None, hsv_lo=lo, hsv_hi=hi)
    det, mask = detect(img, vcfg)
    assert note == "coloured object" and det is not None
    assert mask[10, 10] == 0  # background untouched
    assert lo[1] >= 40         # saturation floor: greys and glare can't match


def test_grey_click_tracks_by_brightness_and_says_so():
    from scrub_arm.vision import VisionCfg

    img = _scene(bg=(60, 120, 190), obj=(40, 40, 40))  # dark grey clamp on orange wood
    lo, hi, note = learn_color(img, 170, 115)
    vcfg = VisionCfg(path=None, hsv_lo=lo, hsv_hi=hi)
    det, mask = detect(img, vcfg)
    assert "brightness" in note
    assert det is not None and abs(det.center[0] - 170) < 3 and mask[10, 10] == 0


def test_huge_blobs_are_treated_as_background():
    from scrub_arm.vision import VisionCfg

    img = _scene(bg=(30, 120, 230), obj=(30, 120, 230))  # the "object" colour is everywhere
    lo, hi, _ = learn_color(img, 170, 115)
    det, _ = detect(img, VisionCfg(path=None, hsv_lo=lo, hsv_hi=hi))
    assert det is None


def test_clamp_is_gripped_across_its_narrow_side(app, config_dir):
    d, kin, cam, vcfg = _calibrated_rig(app, config_dir)
    cam.obj_size = (0.10, 0.015)                       # a 10 cm x 1.5 cm clamp...
    cam.obj_yaw = math.radians(35)                     # ...lying at an angle
    cam.obj_xy = cam.obj_xy + np.array([0.02, 0.015])
    res = find_and_grab(d, kin, cam, vcfg)
    assert res.grab.holding
    jaw_dir, _ = cam.image_axes()                      # jaw_axis_deg = 0: jaws close along image right
    along = np.array([math.cos(cam.obj_yaw), math.sin(cam.obj_yaw), 0.0])
    assert abs(np.dot(jaw_dir, along)) < math.sin(math.radians(8))   # jaws perpendicular to the clamp
    assert np.linalg.norm(kin.fk(d.q_command())[:2, 3] - cam.obj_xy) < 0.005
