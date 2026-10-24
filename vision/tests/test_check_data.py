import cv2
import numpy as np
import pytest

from scrub_vision.check_data import check_labels, check_pair, find_duplicates

CFG = {"min_size": 320, "min_std": 6.0, "min_sharp": 3.0, "min_box": 0.01, "min_box_std": 4.0}


@pytest.mark.parametrize("text,reason", [
    ("0 0.5 0.5 0.2 0.2\n", None),
    ("", "empty label file"),
    ("0 0.5 0.5 0.2\n", "malformed label line"),
    ("99 0.5 0.5 0.2 0.2\n", "class id out of range"),
    ("0 0.95 0.5 0.2 0.2\n", "box outside image"),
    ("0 0.5 0.5 0.001 0.2\n", "box too small"),
    ("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n", "duplicate box"),
])
def test_check_labels(text, reason):
    assert check_labels(text, 16, 0.01)[1] == reason


def _write(tmp_path, img, label="0 0.5 0.5 0.4 0.4\n"):
    cv2.imwrite(str(tmp_path / "a.jpg"), img)
    (tmp_path / "a.txt").write_text(label)
    return check_pair((str(tmp_path / "a.jpg"), str(tmp_path / "a.txt"), 16, CFG))[1]


def test_check_pair_rejects_blank_blurry_and_flat_box(tmp_path):
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)
    assert _write(tmp_path, noisy) is None
    assert _write(tmp_path, np.full((640, 640, 3), 120, np.uint8)) == "image blank/flat"
    ramp = np.tile(np.linspace(0, 255, 640, dtype=np.uint8)[None, :, None], (640, 1, 3))  # contrast but no edges
    assert _write(tmp_path, ramp) == "image too blurry"
    # texture only outside the labeled box -> the box covers flat background
    img = noisy.copy()
    img[160:480, 160:480] = 120
    assert _write(tmp_path, img) == "box over flat region"


def test_find_duplicates_flags_the_later_copy():
    a = np.arange(32, dtype=np.uint8)
    b = a.copy()
    b[0] ^= 1  # 1 bit apart
    c = np.full(32, 255, np.uint8)
    assert find_duplicates({"a.jpg": a, "b.jpg": b, "c.jpg": c}, 6) == {"b.jpg": "near-duplicate of a.jpg"}
