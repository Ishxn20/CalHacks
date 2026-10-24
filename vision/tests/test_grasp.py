import math

import cv2
import numpy as np
import pytest

from scrub_vision import grasp, markers


def scene(angle_deg, bg=30, fg=220, size=(640, 480)):
    """A 200x30 bar rotated by angle_deg (clockwise on screen) on a flat background."""
    img = np.full((size[1], size[0], 3), bg, np.uint8)
    box = cv2.boxPoints(((320, 240), (200, 30), angle_deg))
    cv2.fillPoly(img, [np.int32(box)], (fg, fg, fg))
    return img


@pytest.mark.parametrize("angle", [0, 30, 60, 89, -45, -80])
def test_obb_angle_and_grip_axis(angle):
    (d,) = grasp.segment(scene(angle))
    assert abs(grasp.wrap180(d.long_deg - angle)) < 2
    assert abs(grasp.wrap180(d.grip_deg - (angle + 90))) < 2     # jaws close across the short axis
    assert abs(d.center_px[0] - 320) < 2 and abs(d.center_px[1] - 240) < 2
    assert d.length_px > d.width_px * 5


def test_light_object_on_dark_and_dark_on_light():
    for bg, fg in ((30, 220), (220, 30)):
        (d,) = grasp.segment(scene(20, bg=bg, fg=fg))
        assert abs(grasp.wrap180(d.long_deg - 20)) < 2


def test_empty_and_edge_cut_objects_are_ignored():
    assert grasp.segment(np.full((480, 640, 3), 90, np.uint8)) == []
    img = scene(0)
    cv2.rectangle(img, (0, 200), (60, 260), (255, 255, 255), -1)  # touches the border
    assert len(grasp.segment(img)) == 1


def test_heavy_end_points_to_the_bulky_side():
    img = np.full((480, 640, 3), 30, np.uint8)
    cv2.rectangle(img, (200, 225), (420, 255), (220,) * 3, -1)      # handle, thick
    cv2.rectangle(img, (420, 237), (480, 243), (220,) * 3, -1)      # blade, thin
    (d,) = grasp.segment(img)
    assert d.heavy_dir[0] < -0.9


def test_combine_averages_angles_across_the_wrap():
    dets = []
    for a in (88, -89, 89, -88):
        (d,) = grasp.segment(scene(a))
        dets.append(d)
    out, spread_px, spread_deg = grasp.combine(dets)
    assert abs(abs(out.long_deg) - 90) < 2.5 and spread_deg < 4 and spread_px < 2


def test_refine_uses_the_blob_under_the_yolo_box():
    img = scene(40)
    d = grasp.refine(img, (200, 160, 440, 320), "plug", 0.9)
    assert d.name == "plug" and abs(grasp.wrap180(d.long_deg - 40)) < 2
    assert math.dist(d.center_px, (320, 240)) < 2           # crop offset is added back


def test_markers_roundtrip(tmp_path):
    p = tmp_path / "m.png"
    markers.sheet(str(p), n=4)
    found = markers.detect(cv2.imread(str(p)))
    assert sorted(found) == [0, 1, 2, 3] and all(v["side_px"] > 100 for v in found.values())


def test_roi_ignores_a_busy_table_around_the_mat():
    img = np.random.default_rng(0).integers(40, 200, (480, 640, 3), dtype=np.uint8)   # noisy "wood"
    img[100:400, 150:500] = 230                                                        # plain mat
    box = cv2.boxPoints(((320, 250), (120, 30), 35))
    cv2.fillPoly(img, [np.int32(box)], (40, 40, 40))                                   # tool on the mat
    (d,) = grasp.segment_in(img, (150, 100, 350, 300))
    assert abs(grasp.wrap180(d.long_deg - 35)) < 3 and math.dist(d.center_px, (320, 250)) < 3
