import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from scrub_arm.calibrate import homography_from_points, mm_to_px, px_to_mm  # noqa: E402


def test_homography_round_trip():
    H_true = np.array([[0.5, 0.02, -150.0], [-0.01, 0.48, 90.0], [1e-5, 2e-5, 1.0]])
    px = np.array([[100, 80], [900, 120], [850, 600], [120, 640], [500, 350], [300, 500]], float)
    hom = np.c_[px, np.ones(len(px))] @ H_true.T
    mm = hom[:, :2] / hom[:, 2:]
    H = homography_from_points(px, mm)
    assert np.allclose(px_to_mm(H, px), mm, atol=1e-6)
    assert np.allclose(mm_to_px(H, mm), px, atol=1e-4)


def test_homography_needs_four_points():
    with pytest.raises(ValueError):
        homography_from_points(np.zeros((3, 2)), np.zeros((3, 2)))
