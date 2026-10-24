"""Camera calibration -> repo-root config/camera.yaml.

    python -m scrub_arm.calibrate capture <dir> [--camera 0]      (space = save frame, q = quit)
    python -m scrub_arm.calibrate intrinsics <dir of checkerboard jpgs> [--cols 9 --rows 6 --square-mm 25]
    python -m scrub_arm.calibrate touch [--camera 0]

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


def cmd_touch(args) -> None:
    import cv2

    from .app import build
    from .kinematics import Kinematics

    cam = load_camera()
    app, kin = build(), Kinematics()
    d = app.driver
    d.start()
    pts_mm, zs = [], []
    print("Jog the gripper tip onto each dot (limp / hold / jog <j> <deg>), then `rec`. `done` when finished.")
    while True:
        cmd = input("touch> ").split()
        if not cmd:
            continue
        if cmd[0] == "limp":
            d.limp()
        elif cmd[0] == "hold":
            d.enable()
        elif cmd[0] == "jog":
            q = d.q_command()
            q[int(cmd[1]) - 1] += math.radians(float(cmd[2]))
            d.clear_abort()
            d.move_to(q, speed=0.3)
        elif cmd[0] == "rec":
            q = d.q_measured() if d.enabled else d.read_fresh()[:-1]
            p, _ = kin.fk(q)
            pts_mm.append(p[:2] * 1000)
            zs.append(p[2] * 1000)
            print(f"  dot {len(pts_mm)}: x={p[0]*1000:.1f} y={p[1]*1000:.1f} z={p[2]*1000:.1f} mm")
        elif cmd[0] == "done":
            break
    d.clear_abort()
    d.move_pose("scan", speed=0.5)
    cap = cv2.VideoCapture(args.camera)
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()
    frame = cv2.undistort(frame, cam["K"], cam["dist"])
    clicks = []
    cv2.namedWindow("click dots in order; q when done")
    cv2.setMouseCallback("click dots in order; q when done",
                         lambda e, u, v, *_: clicks.append((u, v)) if e == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        view = frame.copy()
        for i, (u, v) in enumerate(clicks):
            cv2.circle(view, (u, v), 5, (0, 0, 255), -1)
            cv2.putText(view, str(i + 1), (u + 6, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("click dots in order; q when done", view)
        if cv2.waitKey(30) == ord("q"):
            break
    cv2.destroyAllWindows()
    n = min(len(clicks), len(pts_mm))
    H = homography_from_points(np.array(clicks[:n]), np.array(pts_mm[:n]))
    err = np.linalg.norm(px_to_mm(H, clicks[:n]) - np.array(pts_mm[:n]), axis=1)
    h, w = frame.shape[:2]
    cam.update(camera_index=args.camera, scan_pose="scan", H_px_to_mm=H, table_z_mm=float(np.median(zs)),
               gripper_mask_poly_px=cam.get("gripper_mask_poly_px") or [[0, int(h * 0.75)], [w, int(h * 0.75)], [w, h], [0, h]])
    _save(cam)
    print(f"homography from {n} dots, max error {err.max():.1f} mm, table z {np.median(zs):.1f} mm -> {CAMERA_YAML}")
    print("Set pick.table_z_mm in arm/config/arm.yaml to that table z.")
    d.shutdown()


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
    t = sub.add_parser("touch")
    t.add_argument("--camera", type=int, default=0)
    t.set_defaults(fn=cmd_touch)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
