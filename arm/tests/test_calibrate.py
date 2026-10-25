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


def _rot(rx, ry):
    a, b = np.radians(rx), np.radians(ry)
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return Rx @ Ry


def test_charuco_recovers_focal_length():
    """Pinhole views of the board (planar -> a homography each) at varied tilts; calibration finds fx."""
    from scrub_arm.calibrate import charuco_board, charuco_intrinsics, draw_charuco

    W, H, f = 1920, 1080, 800.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    board_img = charuco_board().generateImage((1400, 1000), marginSize=40)
    # board image px -> board units (square = 1): 7 squares across the 1320 px drawn width, origin at margin
    S = np.array([[7 / 1320, 0, -40 * 7 / 1320], [0, 7 / 1320, -40 * 7 / 1320], [0, 0, 1]])
    det, views, rng = cv2.aruco.CharucoDetector(charuco_board()), [], np.random.default_rng(0)
    for _ in range(12):
        R = _rot(rng.uniform(-35, 35), rng.uniform(-35, 35))
        t = np.array([rng.uniform(-3, -1), rng.uniform(-3, -1), rng.uniform(9, 14)])
        Hm = K @ np.c_[R[:, 0], R[:, 1], t] @ S
        frame = cv2.warpPerspective(board_img, Hm, (W, H), borderValue=128)
        corners, ids, _, _ = det.detectBoard(frame)
        if ids is not None and len(ids) >= 8:
            views.append((corners, ids))
            draw_charuco(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), corners, ids)   # the live view's overlay
    rms, K_est, _ = charuco_intrinsics(views, (W, H))
    assert rms < 1.0
    assert abs(K_est[0, 0] - f) / f < 0.03


def test_handeye_recovers_mount_board_and_square_size():
    """Several arm poses viewing a ChArUco board lying on the table -> camera mount (6-DoF, gripper_end frame),
    board pose and the board's square size (a phone screen's scale is unknown), from joint angles + pixels."""
    from scipy.spatial.transform import Rotation as Rot

    from scrub_arm.calibrate import charuco_board, solve_handeye
    from scrub_arm.kinematics import Kinematics

    kin, rng = Kinematics(), np.random.default_rng(1)
    K = np.array([[1087.0, 0, 928.0], [0, 1086.0, 474.0], [0, 0, 1.0]])
    # a mount like the real one: 39 deg tilt, 32 mm sideways (OpenCV camera axes in the tool frame)
    R_m = Rot.from_euler("xyz", [3, 128, 4], degrees=True).as_matrix()
    t_m = np.array([-0.037, -0.032, 0.087])
    s_true, R_b, t_b = 0.0125, Rot.from_euler("z", 20, degrees=True).as_matrix(), np.array([0.30, -0.04, 0.002])
    obj = charuco_board().getChessboardCorners()                    # board frame, in squares
    views = []
    base_q = np.radians([0.0, 92.0, 108.0, -66.0, 0.0, -3.0])        # the real scan pose
    while len(views) < 10:
        q = base_q + np.radians(rng.uniform([-15, -8, -8, -20, -15, -40], [15, 8, 8, 20, 15, 40]))
        p, R = kin.fk(q)
        R_c, t_c = R @ R_m, p + R @ t_m                              # camera in base
        Xc = (R_b @ (s_true * obj).T).T + t_b
        pc = (Xc - t_c) @ R_c                                        # camera coords (OpenCV)
        if np.any(pc[:, 2] <= 0.02):
            continue
        uv = (pc[:, :2] / pc[:, 2:]) * [K[0, 0], K[1, 1]] + K[:2, 2] + rng.normal(0, 0.5, (len(pc), 2))
        ok = (uv[:, 0] > 0) & (uv[:, 0] < 1920) & (uv[:, 1] > 0) & (uv[:, 1] < 1080)
        if ok.sum() >= 12:
            views.append((q, obj[ok], uv[ok]))
    r = solve_handeye(views, K, kin)
    ang = np.degrees(np.arccos(np.clip((np.trace(r["R_tool_cam"].T @ R_m) - 1) / 2, -1, 1)))
    assert np.linalg.norm(r["p_tool_cam"] - t_m) < 1e-3 and ang < 0.3
    assert abs(r["square_m"] - s_true) / s_true < 0.01 and r["rms_px"] < 1.0
    assert r["camera_spread_mm"] > 20                                  # the views were taken from different places
    fixed = solve_handeye(views, K, kin, square_m=s_true)              # a ruler-measured square fixes the scale
    assert fixed["square_m"] == s_true and np.linalg.norm(fixed["p_tool_cam"] - t_m) < 1e-3


def test_handeye_board_is_detected():
    from scrub_arm.calibrate import HANDEYE_SQUARES, charuco_board

    b = charuco_board(HANDEYE_SQUARES)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(b).detectBoard(b.generateImage((1280, 680), marginSize=40))
    assert ids is not None and len(ids) == (HANDEYE_SQUARES[0] - 1) * (HANDEYE_SQUARES[1] - 1)
