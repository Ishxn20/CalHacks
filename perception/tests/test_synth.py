import numpy as np

from scrub_vision.render import CamSpec, pose, render_scene, scene_mesh
from scrub_vision.synth import make_dataset, random_pile
from scrub_vision.templates import classes, load_instruments


def test_dataset_written(tmp_path):
    make_dataset(tmp_path, n=4, seed=0, val_frac=0.25)
    imgs = sorted((tmp_path / "images").rglob("*.jpg"))
    labels = sorted((tmp_path / "labels").rglob("*.txt"))
    assert len(imgs) == 4 and len(labels) == 4
    assert (tmp_path / "data.yaml").exists()
    n_lines = 0
    for lab in labels:
        for line in lab.read_text().splitlines():
            parts = line.split()
            cls, coords = int(parts[0]), np.array(parts[1:], float)
            assert 0 <= cls < len(classes())
            assert len(coords) >= 6 and len(coords) % 2 == 0
            assert coords.min() >= 0 and coords.max() <= 1
            n_lines += 1
    assert n_lines >= 4


def test_occlusion_removes_hidden_pixels():
    insts = load_instruments()
    a, b = scene_mesh(insts["forceps_resano"]), scene_mesh(insts["scissors_potts"])
    cam = CamSpec()
    items = [(a, pose(0, 0, a.extents[2] / 2, 0.0)), (b, pose(0, 0, a.extents[2] + b.extents[2] / 2, 0.3))]
    rgb, masks = render_scene(items, cam, np.random.default_rng(0))
    lower, upper = masks
    assert upper.any() and lower.any()
    assert not (lower & upper).any()
    alone = render_scene(items[:1], cam, np.random.default_rng(0))[1][0]
    assert lower.sum() < alone.sum()        # part of the lower one is hidden


def test_random_pile_stacks_overlapping_items():
    insts = load_instruments()
    pile = random_pile(insts, np.random.default_rng(1), 8)
    zs = [p[1][2, 3] for p in pile]
    assert len(pile) == 8 and max(zs) > min(zs) + 2.0   # something rests on something


def test_gripper_hides_instrument_pixels():
    from scrub_vision.render import add_gripper
    rng = np.random.default_rng(0)
    img = np.full((540, 960, 3), 200, np.uint8)
    out, gm = add_gripper(img, rng)
    assert gm[-1].any() and not gm[: int(540 * 0.7)].any()
    assert (out[gm] < 60).all()
