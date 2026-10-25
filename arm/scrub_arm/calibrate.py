"""Camera calibration -> repo-root config/camera.yaml.

    python -m scrub_arm.calibrate capture <dir> [--camera 0]      (space = save frame, q = quit)
    python -m scrub_arm.calibrate intrinsics <dir of checkerboard jpgs> [--cols 9 --rows 6 --square-mm 25]
    python -m scrub_arm.calibrate touch [--camera 0]
    python -m scrub_arm.calibrate board <png>                      (ChArUco board to show on a phone/tablet)
    python -m scrub_arm.calibrate charuco [--camera 0] [--n 25]  (live view, auto-saves frames, then intrinsics)
    python -m scrub_arm.calibrate handeye [--camera 0]           (camera mount on the gripper, many arm poses)
    python -m scrub_arm.calibrate aim [--camera 0]               (fingertip spot's pixel at 3/6/10 cm + scan)

`charuco` is the forgiving route: a partly hidden or partly out-of-frame board still counts, frames are
saved automatically when the board is held still somewhere new, and green dots show what's covered.

`touch`: jog the gripper tip onto each tape dot (teach-style prompt), FK gives its (x, y) in the base
frame; then at the `scan` pose click the same dots in the undistorted image, in the same order.
The homography maps undistorted pixels at the scan pose -> table (x, y) mm.
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import yaml

CAMERA_YAML = Path(__file__).resolve().parents[2] / "config" / "camera.yaml"


def homography_from_points(px: np.ndarray, mm: np.ndarray) -> np.ndarray:
    import cv2

    px, mm = np.asarray(px, np.float64), np.asarray(mm, np.float64)
    if len(px) < 4 or len(px) != len(mm):
        raise ValueError("need at least 4 matching point pairs")
    H, _ = cv2.findHomography(px, mm, cv2.RANSAC if len(px) > 4 else 0, 3.0)
    if H is None:
        raise ValueError("points are degenerate (collinear?)")
    return H


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, np.float64))
    h = np.c_[pts, np.ones(len(pts))] @ H.T
    return h[:, :2] / h[:, 2:]


def px_to_mm(H: np.ndarray, uv) -> np.ndarray:
    return _apply(H, uv)


def mm_to_px(H: np.ndarray, xy) -> np.ndarray:
    return _apply(np.linalg.inv(H), xy)


def load_camera(path: Path = CAMERA_YAML) -> dict:
    d = yaml.safe_load(path.read_text())
    for k in ("K", "H_px_to_mm"):
        if d.get(k) is not None:
            d[k] = np.array(d[k], float)
    d["dist"] = np.array(d.get("dist") or [0, 0, 0, 0, 0], float)
    return d


def _save(d: dict, path: Path = CAMERA_YAML) -> None:
    out = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    path.write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=None))


def cmd_intrinsics(args) -> None:
    import cv2

    objp = np.zeros((args.rows * args.cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2) * args.square_mm
    obj, img, size = [], [], None
    for f in sorted(glob.glob(str(Path(args.dir) / "*.jpg")) + glob.glob(str(Path(args.dir) / "*.png"))):
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        ok, corners = cv2.findChessboardCorners(g, (args.cols, args.rows))
        if ok:
            obj.append(objp)
            img.append(cv2.cornerSubPix(g, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)))
            size = g.shape[::-1]
    if len(obj) < 8:
        raise SystemExit(f"only {len(obj)} usable checkerboard images; take at least 8 at varied angles")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    d = load_camera() if CAMERA_YAML.exists() else {}
    d.update(width=size[0], height=size[1], K=K, dist=dist.ravel()[:5].tolist(), reproj_rms_px=float(rms))
    _save(d)
    print(f"intrinsics from {len(obj)} images, RMS {rms:.2f} px -> {CAMERA_YAML}")


HANDEYE_SQUARES = (6, 3)   # ~22 mm squares on a phone held sideways: readable from the scan height (7x5: ~13 mm, not)


def charuco_board(squares: tuple = (7, 5)):
    import cv2

    # 7x5 (default) for the lens calibration, held close; HANDEYE_SQUARES for the hand-eye capture, seen from the
    # arm. Square size is relative here: the intrinsics don't depend on it and the hand-eye solve estimates it.
    return cv2.aruco.CharucoBoard(tuple(squares), 1.0, 0.7, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def draw_charuco(view, corners, ids) -> None:
    import cv2

    # OpenCV 5's detectBoard returns (N, 2) corners and (N,) ids; the drawer wants (N, 1, 2) / (N, 1).
    cv2.aruco.drawDetectedCornersCharuco(view, corners.reshape(-1, 1, 2), ids.reshape(-1, 1), (0, 0, 255))


def charuco_intrinsics(views: list, size: tuple) -> tuple:
    """views: [(charuco_corners, charuco_ids)] -> (rms, K, dist)."""
    import cv2

    board, obj, img = charuco_board(), [], []
    for corners, ids in views:
        o, i = board.matchImagePoints(corners, ids)
        if o is not None and len(o) >= 6:
            obj.append(o.astype(np.float32))
            img.append(i.astype(np.float32))
    if len(obj) < 8:
        raise SystemExit(f"only {len(obj)} usable views; need at least 8")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return rms, K, dist


def _handeye_rotation(R_grip: list, R_cam_board: list) -> np.ndarray:
    """Camera axes in the gripper frame (the rotation of AX = XB). For every pose pair, the gripper's relative
    rotation A and the camera's B share an axis up to X (axis(A) = X axis(B)); X is the best rotation mapping
    all B axes onto the A axes (Kabsch). OpenCV 5's Python build dropped calibrateHandEye."""
    from scipy.spatial.transform import Rotation as Rot

    a, b = [], []
    for i in range(len(R_grip)):
        for j in range(i + 1, len(R_grip)):
            alpha = Rot.from_matrix(R_grip[j].T @ R_grip[i]).as_rotvec()
            beta = Rot.from_matrix(R_cam_board[j] @ R_cam_board[i].T).as_rotvec()
            if np.linalg.norm(alpha) > np.radians(2) and np.linalg.norm(beta) > np.radians(2):
                a.append(alpha)
                b.append(beta)
    U, _, Vt = np.linalg.svd(np.array(b).T @ np.array(a))
    return Vt.T @ np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)]) @ U.T


def solve_handeye(views: list, K: np.ndarray, kin, square_m: float | None = None) -> dict:
    """Camera mount on the gripper from several arm poses viewing one ChArUco board lying still on the table.

    views: [(q (6, rad), board points (N, 3) in squares, undistorted pixels (N, 2))]. Unknowns: the mount
    (camera pose in the gripper_end frame, OpenCV axes), the board's pose in the base frame and its square size
    (a phone screen's scale is unknown; the arm's own motion fixes it, or pass a ruler-measured square_m: views that
    mostly rotate the camera leave scale and camera distance tangled). Initialised by per-view PnP, OpenCV's
    hand-eye rotation and a linear solve for the translations + scale, then refined on reprojection error.
    """
    import cv2
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as Rot

    poses = [kin.fk(np.asarray(q, float)) for q, _, _ in views]
    Rcb, ucb = [], []
    for _, obj, px in views:                                   # board -> camera, translation in squares
        ok, rv, tv = cv2.solvePnP(np.asarray(obj, np.float64), np.asarray(px, np.float64), K, None,
                                  flags=cv2.SOLVEPNP_SQPNP)
        Rcb.append(cv2.Rodrigues(rv)[0])
        ucb.append(tv.ravel())
    R_m = _handeye_rotation([R for _, R in poses], Rcb)
    # R_i t_m + s R_i R_m u_i - t_b = -p_i  (the board's origin is the same point seen from every pose)
    A, b = [], []
    for (p, R), u in zip(poses, ucb):
        A.append(np.c_[R, (R @ R_m @ u)[:, None], -np.eye(3)])
        b.append(-p)
    x = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)[0]
    t_m, s, t_b = x[:3], x[3], x[4:]
    if square_m is not None:
        s = square_m
    R_b = poses[0][1] @ R_m @ Rcb[0]

    def unpack(z):
        sq = square_m if square_m is not None else z[12]
        return Rot.from_rotvec(z[:3]).as_matrix(), z[3:6], Rot.from_rotvec(z[6:9]).as_matrix(), z[9:12], sq

    def residuals(z):
        Rm, tm, Rb, tb, sq = unpack(z)
        out = []
        for (p, R), (_, obj, px) in zip(poses, views):
            Rc, tc = R @ Rm, p + R @ tm
            pc = ((Rb @ (sq * np.asarray(obj)).T).T + tb - tc) @ Rc
            out.append(((pc[:, :2] / pc[:, 2:]) * [K[0, 0], K[1, 1]] + K[:2, 2] - px).ravel())
        return np.concatenate(out)

    z0 = np.r_[Rot.from_matrix(R_m).as_rotvec(), t_m, Rot.from_matrix(R_b).as_rotvec(), t_b,
               [] if square_m is not None else [s]]
    sol = least_squares(residuals, z0, loss="soft_l1", f_scale=2.0)
    Rm, tm, Rb, tb, sq = unpack(sol.x)
    res = residuals(sol.x).reshape(-1, 2)
    per_view, i = [], 0
    for _, obj, _ in views:
        per_view.append(float(np.sqrt(np.mean(np.sum(res[i:i + len(obj)] ** 2, 1)))))
        i += len(obj)
    cams = np.array([p + R @ tm for p, R in poses])
    return {"R_tool_cam": Rm, "p_tool_cam": tm, "R_base_board": Rb, "t_base_board": tb, "square_m": float(sq),
            "rms_px": float(np.sqrt(np.mean(np.sum(res ** 2, 1)))), "per_view_rms_px": per_view,
            # how far apart the camera was across the views: a few cm of rotation-only motion can't fix the distance
            "camera_spread_mm": float(np.ptp(cams, axis=0).max() * 1000)}


def cmd_board(args) -> None:
    import cv2

    sq = tuple(args.squares)
    img = charuco_board(sq).generateImage((200 * sq[0] + 80, 200 * sq[1] + 80), marginSize=40)
    cv2.imwrite(args.png, img)
    print(f"wrote {args.png}: open it full screen on a phone/tablet held sideways, brightness up")


def cmd_charuco(args) -> None:
    import cv2

    det, out = cv2.aruco.CharucoDetector(charuco_board()), Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    views, centres, covered, prev, still, size = [], [], [], None, 0, None
    while len(views) < args.n:
        ok, frame = cap.read()
        if not ok:
            continue
        size = frame.shape[1::-1]
        corners, ids, _, _ = det.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        view = frame.copy()
        for p in covered:
            cv2.circle(view, p, 6, (0, 200, 0), -1)
        n = 0 if ids is None else len(ids)
        msg = "show the board"
        if n >= 8:
            draw_charuco(view, corners, ids)
            pts = corners.reshape(-1, 2)
            # Held still (corners barely moved since last frame) and somewhere new -> save.
            still = still + 1 if prev is not None and len(prev) == len(pts) and \
                np.abs(prev - pts).mean() < 2.0 else 0
            c = pts.mean(0)
            new = all(np.linalg.norm(c - q) > 0.08 * size[0] for q in centres)
            msg = "hold still..." if new else "move somewhere new (or tilt)"
            if still >= 4 and new:
                cv2.imwrite(str(out / f"{len(views):03d}.jpg"), frame)
                views.append((corners, ids))
                centres.append(c)
                covered += [tuple(int(v) for v in p) for p in pts]
                still = 0
            prev = pts
        else:
            prev, still = None, 0
        cv2.putText(view, f"{len(views)}/{args.n} saved  {n} corners  {msg}  (q: finish)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("charuco calibration", cv2.resize(view, None, fx=0.5, fy=0.5))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    rms, K, dist = charuco_intrinsics(views, size)
    d = load_camera() if CAMERA_YAML.exists() else {}
    d.update(width=size[0], height=size[1], K=K, dist=dist.ravel()[:5].tolist(), reproj_rms_px=float(rms))
    _save(d)
    print(f"intrinsics from {len(views)} views, RMS {rms:.2f} px, fx {K[0, 0]:.0f} fy {K[1, 1]:.0f} -> {CAMERA_YAML}")


def cmd_capture(args) -> None:
    import cv2

    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    cap, n = cv2.VideoCapture(args.camera), len(list(out.glob("*.jpg")))
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        view = frame.copy()
        cv2.putText(view, f"{n} saved - space: save, q: quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("capture", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            cv2.imwrite(str(out / f"{n:03d}.jpg"), frame)
            n += 1
        elif k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f"{n} images in {out}")


def _click_points(frame, title: str) -> list:
    import cv2

    clicks = []
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, lambda e, u, v, *_: clicks.append((u, v)) if e == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        view = frame.copy()
        for i, (u, v) in enumerate(clicks):
            cv2.circle(view, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (u + 6, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(title, view)
        if cv2.waitKey(30) == ord("q"):
            break
    cv2.destroyAllWindows()
    return clicks


ARM_COMMANDS = "s | limp | hold | jog <joint> <deg> | set <joint> <deg> | go <pose> [speed] | open | close | reset"


def _arm_command(d, cmd: list) -> bool:
    """The arm controls shared by the touch and hand-eye tools. True if cmd was one of them."""
    if cmd[0] == "limp":
        d.limp()
    elif cmd[0] == "hold":
        d.enable()
    elif cmd[0] in ("jog", "set"):
        q = d.q_command()
        q[int(cmd[1]) - 1] = (q[int(cmd[1]) - 1] if cmd[0] == "jog" else 0.0) + math.radians(float(cmd[2]))
        d.clear_abort()
        d.move_to(q, speed=0.3)
    elif cmd[0] == "go" and len(cmd) > 1:
        d.clear_abort()
        d.move_pose(cmd[1], speed=float(cmd[2]) if len(cmd) > 2 else 0.5)
        print(f"  at {cmd[1]}")
    elif cmd[0] in ("open", "close"):
        d.clear_abort()
        d.gripper_open() if cmd[0] == "open" else d.gripper_grasp()   # close = the soft squeeze
    elif cmd[0] == "reset":
        d.clear_fault()                          # after the collision guard froze the arm
        print("  fault cleared")
    elif cmd[0] in ("s", "status"):
        q = d.q_measured() if d.enabled else d.read_fresh()[:-1]
        print("  joints (deg): " + " ".join(f"{math.degrees(v):+.1f}" for v in q) + f"  enabled={d.enabled} fault={d.fault}")
    else:
        return False
    return True


def cmd_handeye(args) -> None:
    """Interactive multi-view hand-eye capture: board flat on the table, arm in many poses, `snap` each."""
    import json
    import time

    import cv2

    from .app import build
    from .kinematics import Kinematics
    from .teach import park_and_shutdown

    cam = load_camera()
    K, dist = cam["K"], cam["dist"]
    board = charuco_board(HANDEYE_SQUARES)
    det = cv2.aruco.CharucoDetector(board)
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    square_m = args.square_mm / 1000 if args.square_mm else None
    saved = out / "views.json"
    old = [(np.array(v["q_rad"]), np.array(v["obj"]), np.array(v["px"])) for v in json.loads(saved.read_text())] \
        if (args.append or args.offline) and saved.exists() else []
    if args.offline:                         # re-solve the saved views; no arm, no camera
        r = solve_handeye(old, K, Kinematics(), square_m)
        print(f"{len(old)} views: RMS {r['rms_px']:.2f} px, camera {np.round(r['p_tool_cam'] * 1000, 1).tolist()} mm, "
              f"squares {r['square_m'] * 1000:.2f} mm, camera moved {r['camera_spread_mm']:.0f} mm across the views")
        return
    app, kin = build(), Kinematics()
    d = app.driver
    d.start()
    cap = cv2.VideoCapture(args.camera)
    views, result, cmd = list(old), None, []
    if views:
        print(f"  continuing with {len(views)} saved views from {saved}")
    print(f"The {HANDEYE_SQUARES[0]}x{HANDEYE_SQUARES[1]} board (hw_check/handeye_board.png) flat on the mat, fully in view. For each of 8-12 poses: move the arm (hold + jog/set/go, or limp,\n"
          "place it, hold), then `snap`. Vary everything: height, left/right, tilt (jog 4 / jog 5) and wrist roll\n"
          "(jog 6 +-30-60). `solve` to check, `drop` to discard the last view, `done` to save and park.")
    try:
        while True:
            cmd = input("handeye> ").split()
            if not cmd:
                continue
            try:
                if _arm_command(d, cmd):
                    pass
                elif cmd[0] == "snap":
                    if not d.enabled:
                        print("  `hold` first: the joints are read while the arm holds still")
                        continue
                    time.sleep(0.5)                                  # let the arm and the image settle
                    for _ in range(6):
                        cap.read()                                   # drop frames buffered while it moved
                    ok, frame = cap.read()
                    if not ok:
                        print("  no camera frame")
                        continue
                    frame = cv2.undistort(frame, K, dist)
                    corners, ids, _, _ = det.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    n = 0 if ids is None else len(ids)
                    if n < 6:
                        print(f"  only {n} board corners in view (need 6+): move so more of the board shows")
                        continue
                    obj, img = board.matchImagePoints(corners, ids)
                    q = d.q_measured()
                    views.append((q, obj.reshape(-1, 3), img.reshape(-1, 2)))
                    k = len(views)
                    cv2.imwrite(str(out / f"view_{k:02d}.jpg"), frame)
                    saved.write_text(json.dumps([{"q_rad": list(map(float, q_)), "obj": o.tolist(), "px": x.tolist()}
                                                 for q_, o, x in views]))       # kept even if the session dies
                    print(f"  view {k}: {n} corners, joints " + " ".join(f"{math.degrees(v):+.1f}" for v in q))
                elif cmd[0] == "drop":
                    if views:
                        views.pop()
                    print(f"  {len(views)} views kept")
                elif cmd[0] in ("solve", "done"):
                    if len(views) < 6:
                        print(f"  {len(views)} views; need at least 6 (8-12 is better)")
                        continue
                    result = solve_handeye(views, K, kin, square_m)
                    p, R = result["p_tool_cam"], result["R_tool_cam"]
                    tilt = math.degrees(math.acos(np.clip(R[:, 2] @ kin.tool_axis_local, -1, 1)))
                    print(f"  RMS {result['rms_px']:.2f} px over {len(views)} views (worst view "
                          f"{int(np.argmax(result['per_view_rms_px'])) + 1}: {max(result['per_view_rms_px']):.2f} px); "
                          f"camera at ({p[0]*1000:.1f}, {p[1]*1000:.1f}, {p[2]*1000:.1f}) mm in the tool frame, optical "
                          f"axis {tilt:.1f} deg off the tool axis; board squares {result['square_m']*1000:.2f} mm; "
                          f"camera moved {result['camera_spread_mm']:.0f} mm across the views")
                    if result["camera_spread_mm"] < 60:
                        print("  WARNING: the views were taken from nearly one place, so the camera's distance is poorly "
                              "fixed. Add views from other positions (limp, aim the camera at the board from the side / "
                              "closer / higher, hold, snap)" + ("" if square_m else ", and pass --square-mm"))
                    if cmd[0] == "done":
                        break
                else:
                    print(f"  commands: {ARM_COMMANDS} | snap | drop | solve | done")
            except Exception as ex:  # noqa: BLE001 - a typo must not kill the session
                print(f"  error: {type(ex).__name__}: {ex}")
    finally:
        cap.release()
        (out / "views.json").write_text(json.dumps([{"q_rad": list(map(float, q)), "obj": o.tolist(), "px": x.tolist()}
                                                    for q, o, x in views]))
        if result is not None and cmd and cmd[0] == "done":
            cam["handeye"] = {"p_tool_cam_m": result["p_tool_cam"].round(5).tolist(),
                              "R_tool_cam": result["R_tool_cam"].round(6).tolist(),
                              "rms_px": round(result["rms_px"], 3), "n_views": len(views),
                              "square_m": round(result["square_m"], 6),
                              "camera_spread_mm": round(result["camera_spread_mm"], 1)}
            _save(cam)
            print(f"  saved the camera mount to {CAMERA_YAML} (views in {out})")
        park_and_shutdown(app)


def cmd_touch(args) -> None:
    import cv2

    from .app import build
    from .kinematics import Kinematics
    from .teach import park_and_shutdown

    cam = load_camera()
    app, kin = build(), Kinematics()
    d = app.driver
    d.start()
    pts_mm, zs, pts_q = [], [], []
    print("Jog the gripper tip onto each dot (limp / hold / jog <j> <deg> / go <pose>), then `rec`. `done` when finished.")
    while True:
        cmd = input("touch> ").split()
        if not cmd:
            continue
        try:
            if _arm_command(d, cmd):
                pass
            elif cmd[0] == "rec":
                q = d.q_measured() if d.enabled else d.read_fresh()[:-1]
                p, _ = kin.fk(q)
                pts_mm.append(p[:2] * 1000)
                zs.append(p[2] * 1000)
                pts_q.append([round(math.degrees(v), 3) for v in q])
                print(f"  dot {len(pts_mm)}: x={p[0]*1000:.1f} y={p[1]*1000:.1f} z={p[2]*1000:.1f} mm")
            elif cmd[0] == "done":
                break
            else:
                print(f"  commands: {ARM_COMMANDS} | rec | done")
        except Exception as ex:  # noqa: BLE001 - a typo (e.g. go before hold) must not kill the session
            print(f"  error: {type(ex).__name__}: {ex}")
    d.clear_abort()
    d.move_pose("scan", speed=0.5)
    d.gripper_open()                     # fingers out of the way of the dots
    cap = cv2.VideoCapture(args.camera)
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()
    frame = cv2.undistort(frame, cam["K"], cam["dist"])
    clicks = _click_points(frame, "click dots in order; q when done")
    n = min(len(clicks), len(pts_mm))
    H = homography_from_points(np.array(clicks[:n]), np.array(pts_mm[:n]))
    err = np.linalg.norm(px_to_mm(H, clicks[:n]) - np.array(pts_mm[:n]), axis=1)
    h, w = frame.shape[:2]
    # The work mat: the perception service ignores detections mostly off it (e.g. wood grain).
    mat_px = _click_points(frame, "click the work mat's 4 corners (q to skip)")
    if len(mat_px) >= 3:
        cam["mat_mm"] = px_to_mm(H, mat_px).round(1).tolist()
    cam.update(camera_index=args.camera, scan_pose="scan", H_px_to_mm=H, table_z_mm=float(np.median(zs)),
               gripper_mask_poly_px=cam.get("gripper_mask_poly_px") or [[0, int(h * 0.75)], [w, int(h * 0.75)], [w, h], [0, h]])
    # Raw correspondences (joint angles at each dot + its click at the scan pose) so the camera mount on the
    # gripper can be solved later (PnP at the scan pose).
    cam["touch_points"] = [{"q_deg": qd, "px": [int(u), int(v)]} for qd, (u, v) in zip(pts_q[:n], clicks[:n])]
    cam["scan_q_deg"] = [round(math.degrees(v), 3) for v in d.q_measured()]
    _save(cam)
    print(f"homography from {n} dots, max error {err.max():.1f} mm, table z {np.median(zs):.1f} mm -> {CAMERA_YAML}")
    print("Set pick.table_z_mm in arm/config/arm.yaml to that table z.")
    park_and_shutdown(app)               # never just cut torque at scan: the arm would drop


AIM_STAGES_CM = (3, 6, 10)               # sim/center.py STAGES_M


def cmd_aim(args) -> None:
    """Measure once where the spot between the open fingertips shows up in the wrist camera: at each centring
    stage height (the pixel the target must sit on so the straight-down descent lands the jaws on it) and at the
    scan pose (where the fingertips really were vs where the scan homography puts that spot: the plan's offset)."""
    import cv2

    from .app import build
    from .kinematics import Kinematics
    from .teach import park_and_shutdown

    cam = load_camera()
    app, kin = build(), Kinematics()
    d, asst, table_z = app.driver, app.assistant, app.cfg.pick.table_z
    d.start()
    print("Put a small dot on the mat. `hold`, `go scan`, `mz -200`, `down` (the pick orientation: arm.yaml "
          "pick.tilt_pose), `open`, then bring the fingertips onto the dot, midway between them, touching the mat "
          f"(mx | my | mz <mm> | down | {ARM_COMMANDS}), then `rec`.")
    kin = asst.kin                           # carries the pick tilt

    def off_pick_deg(q):                     # how far the gripper is from the pick orientation at its jaw yaw
        _, R = kin.fk(q)
        jaw = R @ kin.jaw_axis_local
        E = kin._target_rotation(math.atan2(jaw[1], jaw[0]))
        return math.degrees(np.linalg.norm(kin.pin.log3(R.T @ E)))
    while True:
        cmd = input("aim> ").split()
        if not cmd:
            continue
        try:
            if _arm_command(d, cmd):
                pass
            elif cmd[0] in ("mx", "my", "mz"):
                d.clear_abort()
                asst.move_line(*(float(cmd[1]) if cmd[0] == ax else 0.0 for ax in ("mx", "my", "mz")))
                p, _ = kin.fk(d.q_measured())
                print(f"  tool at ({p[0]*1000:.1f}, {p[1]*1000:.1f}, {p[2]*1000:.1f}) mm")
            elif cmd[0] == "down":
                p, R = kin.fk(d.q_measured())
                if p[2] - table_z < 0.08:            # the wrist swings through a big arc: not near the mat
                    print("  go up first (mz) to 80+ mm above the mat")
                    continue
                jaw = R @ kin.jaw_axis_local
                d.clear_abort()
                d.move_to(asst._solve(p[0] * 1000, p[1] * 1000, p[2], math.degrees(math.atan2(jaw[1], jaw[0]))),
                          speed=0.3, exact=True)
                print(f"  in the pick orientation ({off_pick_deg(d.q_measured()):.1f} deg off)")
            elif cmd[0] == "rec":
                p, _ = kin.fk(d.q_measured())
                if off_pick_deg(d.q_measured()) > 5:  # the aim pixels hold for the pick orientation only
                    print(f"  {off_pick_deg(d.q_measured()):.0f} deg off the pick orientation: go up 80+ mm, `down`")
                    continue
                if p[2] - table_z > 0.005:       # first real run: rec at 11.6 mm, ~1 cm above the dot
                    print(f"  tool z {p[2]*1000:.1f} mm: lower it onto the mat (table z {table_z*1000:.1f} mm), "
                          "mz -2 at a time")
                    continue
                break
            else:
                print(f"  commands: mx | my | mz <mm> | down | {ARM_COMMANDS} | rec")
        except Exception as ex:  # noqa: BLE001 - a typo must not kill the session
            print(f"  error: {type(ex).__name__}: {ex}")
    p_dot, _ = kin.fk(d.q_measured())
    print(f"  dot at x={p_dot[0]*1000:.1f} y={p_dot[1]*1000:.1f} mm (tool z {p_dot[2]*1000:.1f} mm)")
    cap = cv2.VideoCapture(args.camera)
    shots = Path("hw_check/aim")
    shots.mkdir(parents=True, exist_ok=True)

    def click(name, title):
        for _ in range(10):                  # drain stale frames
            ok, frame = cap.read()
        frame = cv2.undistort(frame, cam["K"], cam["dist"])
        clicks = _click_points(frame, title)
        if not clicks:
            raise SystemExit("no click: nothing saved")
        u, v = (int(c) for c in clicks[-1])
        cv2.drawMarker(frame, (u, v), (255, 0, 255), cv2.MARKER_CROSS, 60, 4)
        cv2.imwrite(str(shots / f"{name}.jpg"), frame)      # to check the click afterwards
        return [u, v]

    d.gripper_open()                         # the stages look past OPEN jaws (closed ones hide the dot)
    aim = {}
    for h in AIM_STAGES_CM:                  # straight up off the dot: it stays put
        p, _ = kin.fk(d.q_measured())
        asst.move_line(0, 0, (table_z + h / 100 - p[2]) * 1000)
        aim[h] = click(f"{h}cm", f"{h} cm: click the PEN DOT on the mat, not the fingertips (last click counts), q")
    d.move_pose("scan", speed=0.5)
    d.gripper_open()                         # as at a real scan
    scan_px = click("scan", "scan pose: click the PEN DOT on the mat (last click counts), then q")
    cap.release()
    off = p_dot[:2] * 1000 - px_to_mm(cam["H_px_to_mm"], [scan_px])[0]
    cam["aim_px"] = aim
    cam["scan_aim"] = {"px": scan_px, "dot_mm": (p_dot[:2] * 1000).round(1).tolist(), "offset_mm": off.round(1).tolist()}
    _save(cam)
    print(f"aim pixels {aim}; scan homography is off by ({off[0]:+.1f}, {off[1]:+.1f}) mm -> {CAMERA_YAML}")
    park_and_shutdown(app)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="scrub_arm.calibrate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("dir")
    c.add_argument("--camera", type=int, default=0)
    c.set_defaults(fn=cmd_capture)
    i = sub.add_parser("intrinsics")
    i.add_argument("dir")
    i.add_argument("--cols", type=int, default=9)
    i.add_argument("--rows", type=int, default=6)
    i.add_argument("--square-mm", type=float, default=25.0)
    i.set_defaults(fn=cmd_intrinsics)
    b = sub.add_parser("board")
    b.add_argument("png")
    b.add_argument("--squares", type=int, nargs=2, default=[7, 5], help=f"hand-eye: {HANDEYE_SQUARES[0]} {HANDEYE_SQUARES[1]}")
    b.set_defaults(fn=cmd_board)
    ch = sub.add_parser("charuco")
    ch.add_argument("--camera", type=int, default=0)
    ch.add_argument("--n", type=int, default=25)
    ch.add_argument("--dir", default=str(Path(__file__).resolve().parents[2] / "hw_check" / "charuco"))
    ch.set_defaults(fn=cmd_charuco)
    he = sub.add_parser("handeye")
    he.add_argument("--camera", type=int, default=0)
    he.add_argument("--dir", default=str(Path(__file__).resolve().parents[2] / "hw_check" / "handeye"))
    he.add_argument("--square-mm", type=float, help="the board's square size measured with a ruler (mm)")
    he.add_argument("--append", action="store_true", help="keep the views already saved in --dir and add more")
    he.add_argument("--offline", action="store_true", help="re-solve the saved views (no arm, no camera)")
    he.set_defaults(fn=cmd_handeye)
    t = sub.add_parser("touch")
    t.add_argument("--camera", type=int, default=0)
    t.set_defaults(fn=cmd_touch)
    a = sub.add_parser("aim")
    a.add_argument("--camera", type=int, default=0)
    a.set_defaults(fn=cmd_aim)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
