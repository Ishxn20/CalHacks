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


def test_rasterize_overlapping_triangles_is_union():
    from scrub_vision.templates import rasterize_xy
    v = np.array([[0, 0], [40, 0], [0, 40], [10, 10], [50, 10], [10, 50]], float)
    m = rasterize_xy(v, np.array([[0, 1, 2], [3, 4, 5]]), (60, 60))
    assert m[15, 15] == 255      # inside both triangles: must stay filled


def test_template_mask_is_solid():
    from scrub_vision.templates import build_templates, load_instruments
    t = next(t for t in build_templates(load_instruments(), 0.5) if t.fine_id == "forceps_resano")
    import cv2
    _, hier = cv2.findContours((t.mask > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    holes = sum(h[3] != -1 for h in hier[0])
    assert holes <= 1            # only the gap between the tines


def test_instrument_colours_vary_beyond_green():
    import cv2
    insts = load_instruments()
    m = scene_mesh(insts["clamp_side_biting"])
    hues, vals = [], []
    for seed in range(40):
        bgr, (mask,) = render_scene([(m, pose(0, 0, m.extents[2] / 2, 0.0))], CamSpec(), np.random.default_rng(seed))
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[mask]
        hues.append(np.median(hsv[:, 0]))
        vals.append(np.median(hsv[:, 2]))
    hues, vals = np.array(hues), np.array(vals)
    assert ((hues < 35) | (hues > 95)).sum() >= 10      # plenty of non-green instruments
    assert vals.min() < 90 and vals.max() > 180           # dark and light ones too


def test_instances_in_one_scene_can_differ_in_colour():
    insts = load_instruments()
    m = scene_mesh(insts["forceps_resano"])
    differ = 0
    for seed in range(10):
        bgr, (a, b) = render_scene([(m, pose(-80, 0, 5, 0.0)), (m, pose(80, 0, 5, 0.0))], CamSpec(),
                                   np.random.default_rng(seed))
        differ += np.abs(np.median(bgr[a], 0) - np.median(bgr[b], 0)).max() > 30
    assert differ >= 5
