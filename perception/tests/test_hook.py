import numpy as np
import pytest

from scrub_vision.hook import IDS, hook_type
from scrub_vision.render import CamSpec, pose, render_scene, scene_mesh
from scrub_vision.templates import load_instruments


@pytest.mark.parametrize("fid", IDS)
@pytest.mark.parametrize("yaw,flip", [(0.4, False), (2.6, True)])
def test_hook_tells_the_micro_scissors_apart(fid, yaw, flip):
    """Lying on their side at the 10 cm hover's scale: forward (straight 45 deg bend) vs back (curled hook)."""
    cam = CamSpec(w=1920, h=1080, fx=1087.0, height_mm=180.0)
    m = scene_mesh(load_instruments()[fid], "side")
    _, (mask,) = render_scene([(m, pose(10.0, 5.0, m.extents[2] / 2, yaw, flip=flip))], cam, np.random.default_rng(1))
    got, scores = hook_type(mask)
    assert got == fid, scores


def test_no_answer_when_the_tip_is_cut_off():
    cam = CamSpec(w=1920, h=1080, fx=1087.0, height_mm=180.0)
    m = scene_mesh(load_instruments()["scissors_micro_back"], "side")
    _, (mask,) = render_scene([(m, pose(0.0, 0.0, m.extents[2] / 2, 0.0))], cam, np.random.default_rng(1))
    cols = np.nonzero(mask.any(0))[0]
    mask[:, cols.max() - 60:] = False                                     # tip end hidden (e.g. by the gripper)
    blocked = np.zeros_like(mask)
    blocked[:, cols.max() - 60:] = True
    assert hook_type(mask, blocked=blocked)[0] is None
