import numpy as np
import pytest
from scrub_arm.config import load_tray_config

from scrub_vision import render_dataset as rd


def test_every_tray_instrument_has_a_mesh_recipe():
    # a new tool in tray.yaml without a recipe must fail loudly, not silently drop out of training
    ids = list(load_tray_config().instruments)
    known = set(rd.MESHES) | set(rd.BLADES) | {"probe_coronary"}
    assert set(ids) <= known


def test_unknown_instrument_raises():
    with pytest.raises(KeyError, match="no mesh recipe"):
        rd.build_pool(["scalpel_99"])


@pytest.mark.skipif(not rd.RAW.exists(), reason="STLs missing from vision/assets/stl")
def test_scene_labels_are_valid_yolo_boxes():
    ids = ["scalpel_11", "scalpel_20", "probe_coronary"]
    img, labels = rd.render_scene(rd.build_pool(ids), ids, np.random.default_rng(0))
    assert img.shape == (rd.SIZE, rd.SIZE, 3) and labels
    for c, x, y, w, h in labels:
        assert 0 <= c < len(ids) and 0 < w <= 1 and 0 < h <= 1
        assert x - w / 2 >= 0 and x + w / 2 <= 1 and y - h / 2 >= 0 and y + h / 2 <= 1
