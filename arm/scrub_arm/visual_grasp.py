"""Find an object with the wrist camera, centre over it, and grab it.

Image-based visual servoing, so no camera intrinsics or hand-eye calibration are needed:

calibrate_camera(): with the object directly under the jaws (as for a successful `grab`),
    remember where it appears in the image ("target pixel") and the hover height, then make
    two 2 cm test moves to learn how tool motion shifts it in the image (a 2x2 Jacobian).

find_and_grab(): go to the calibrated hover height pointing down, then repeatedly
    detect the object and step sideways by J^-1 * (target - detected) until it sits on the
    target pixel, then run grab_below().

The Jacobian is stored in the tool's frame, so it still works if the wrist ends up rotated.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from .driver import ArmDriver
from .grasp import GrabError, GrabResult, down_seeds, grab_below, point_down, tilt_from_down
from .kinematics import IKError, Kinematics
from .vision import VisionCfg, detect, tool_yaw

log = logging.getLogger(__name__)

TEST_MOVE_M = 0.02
CENTERED_PX = 6.0
MAX_STEP_M = 0.04
SERVO_SPEED = 0.04   # m/s
GAIN = 0.8


@dataclass
class SeekResult:
    grab: GrabResult
    steps: int
    final_error_px: float

    def __str__(self) -> str:
        return f"centred in {self.steps} step(s) ({self.final_error_px:.1f} px off); {self.grab}"


def _rot2(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def locate(cam, vcfg: VisionCfg, frames: int = 3) -> np.ndarray | None:
    """Median object centre over a few frames taken after now. None if not seen in most of them."""
    pts = []
    for _ in range(frames):
        det, _ = detect(cam.fresh(), vcfg)
        if det is not None:
            pts.append(det.center)
    if len(pts) * 2 <= frames:
        return None
    return np.median(np.array(pts), axis=0)


def _move_xy(driver: ArmDriver, kin: Kinematics, dxy: np.ndarray) -> None:
    q = driver.q_command()
    try:
        path = kin.line(q, np.array([dxy[0], dxy[1], 0.0]))
    except IKError as e:
        raise GrabError(f"can't move sideways {np.linalg.norm(dxy) * 1000:.0f} mm from here: {e}") from e
    driver.follow_path(path, duration=max(0.3, float(np.linalg.norm(dxy)) / SERVO_SPEED))
    driver.sleep(0.15)  # let the image settle


def _move_to_height(driver: ArmDriver, kin: Kinematics, z: float) -> None:
    q = driver.q_command()
    dz = z - float(kin.fk(q)[2, 3])
    if abs(dz) < 0.001:
        return
    try:
        path = kin.line(q, np.array([0.0, 0.0, dz]))
    except IKError as e:
        raise GrabError(f"can't reach the hover height ({z * 1000:.0f} mm): {e}") from e
    driver.follow_path(path, duration=max(0.4, abs(dz) / driver.cfg.grab.speed / 2))


def calibrate_camera(driver: ArmDriver, kin: Kinematics, cam, vcfg: VisionCfg) -> str:
    """Run with the gripper pointing down and the object directly under the jaws."""
    table_z = driver.cfg.grab.table_z
    if table_z is None:
        raise GrabError("record the table height first (`table` in the teach tool)")
    if vcfg.hsv_lo is None:
        raise GrabError("click the object in the camera window first so I know its colour")
    q = driver.q_command()
    if tilt_from_down(kin, q) > math.radians(3):
        raise GrabError("point the gripper straight down first (`d`)")

    p0 = locate(cam, vcfg)
    if p0 is None:
        raise GrabError("can't see the object; click it in the camera window to (re)learn its colour")
    cols = []
    for axis in (0, 1):
        step = np.zeros(2)
        step[axis] = TEST_MOVE_M
        _move_xy(driver, kin, step)
        p = locate(cam, vcfg)
        _move_xy(driver, kin, -step)
        if p is None:
            raise GrabError("lost the object during the test move; start with it nearer the image centre")
        cols.append((p - p0) / TEST_MOVE_M)
    J_world = np.column_stack(cols)                     # px per metre of base-frame x / y
    if abs(np.linalg.det(J_world)) < 1e3 or np.min(np.linalg.norm(J_world, axis=0)) < 100:
        raise GrabError("the object barely moved in the image during the test moves; is the camera seeing it?")

    yaw = tool_yaw(kin, q)
    h, w = cam.fresh().shape[:2]
    vcfg.image_size = [int(w), int(h)]
    vcfg.target_px = [float(p0[0]), float(p0[1])]
    vcfg.jacobian_tool = (J_world @ _rot2(yaw)).tolist()  # J_world = J_tool @ R(yaw)^T
    vcfg.hover_m = float(kin.fk(q)[2, 3]) - table_z
    vcfg.save()
    px_per_cm = np.linalg.norm(J_world, axis=0).mean() / 100
    return (f"calibrated: target pixel ({p0[0]:.0f}, {p0[1]:.0f}), hover {vcfg.hover_m * 1000:.0f} mm "
            f"above the table, ~{px_per_cm:.0f} px per cm")


def _check_ready(driver: ArmDriver, vcfg: VisionCfg) -> float:
    table_z = driver.cfg.grab.table_z
    if table_z is None:
        raise GrabError("record the table height first (`t` / `table`)")
    if vcfg.hsv_lo is None:
        raise GrabError("click the object in the camera window first so I know its colour")
    if not vcfg.calibrated:
        raise GrabError("calibrate the camera first (`c` with the object under the jaws)")
    return table_z


def _centre(driver: ArmDriver, kin: Kinematics, cam, vcfg: VisionCfg, max_steps: int = 12) -> tuple[int, float]:
    """Step sideways until the object sits on the target pixel. Returns (steps, final error px)."""
    target = np.array(vcfg.target_px)
    J_tool = np.array(vcfg.jacobian_tool)
    for step in range(max_steps + 1):
        p = locate(cam, vcfg)
        if p is None:
            raise GrabError("lost sight of the object while centring over it")
        err = target - p
        log.info("servo step %d: object at (%.0f, %.0f), %.1f px off", step, p[0], p[1], np.linalg.norm(err))
        if np.linalg.norm(err) <= CENTERED_PX:
            return step, float(np.linalg.norm(err))
        J = J_tool @ _rot2(tool_yaw(kin, driver.q_command())).T
        dxy = GAIN * np.linalg.solve(J, err)
        if np.linalg.norm(dxy) > MAX_STEP_M:
            dxy *= MAX_STEP_M / np.linalg.norm(dxy)
        _move_xy(driver, kin, dxy)
    raise GrabError(f"couldn't centre on the object ({np.linalg.norm(err):.0f} px off after {max_steps} steps)")


ALIGN_MIN_ASPECT = 1.6          # objects longer than this ratio get the jaws turned across them
ALIGN_TEST_DEG = 10.0


def _wrap90(deg: float) -> float:
    """Wrap an angle between undirected lines into (-90, 90]."""
    return -((-deg + 90.0) % 180.0 - 90.0)


def _object_angle(cam, vcfg: VisionCfg, frames: int = 3) -> tuple[float, float] | None:
    """(long-axis image angle in deg, aspect) averaged over a few fresh frames, or None if not seen."""
    vecs, aspects = [], []
    for _ in range(frames):
        det, _ = detect(cam.fresh(), vcfg)
        if det is not None:
            t = math.radians(2 * det.angle_deg)  # doubled so 0 and 180 deg (the same line) average properly
            vecs.append((math.cos(t), math.sin(t)))
            aspects.append(det.aspect)
    if len(vecs) * 2 <= frames:
        return None
    c, s_ = np.mean(vecs, axis=0)
    return math.degrees(math.atan2(s_, c)) / 2 % 180.0, float(np.median(aspects))


def _align_jaws(driver: ArmDriver, kin: Kinematics, cam, vcfg: VisionCfg) -> float:
    """Turn the wrist (joint6) so the jaws close across a long object's narrow side.
    Rotating joint6 spins the gripper about its own axis, so the tool point doesn't move.
    Returns the rotation made (deg); 0 for round objects."""
    seen = _object_angle(cam, vcfg)
    if seen is None or seen[1] < ALIGN_MIN_ASPECT:
        return 0.0
    want = vcfg.jaw_axis_deg + 90.0          # long axis should end up perpendicular to the jaw axis
    if abs(_wrap90(want - seen[0])) < 5.0:
        return 0.0
    j6 = driver.cfg.joints[5]

    def turn(deg: float) -> None:
        q = driver.q_command()
        q[5] += math.radians(deg)
        if not j6.lower <= q[5] <= j6.upper:   # a line looks the same turned 180 deg
            q[5] -= math.copysign(math.pi, deg)
        driver.move_to(q, speed=0.5)
        driver.sleep(0.15)

    turn(ALIGN_TEST_DEG)                       # which way does the image turn when the wrist turns?
    after = _object_angle(cam, vcfg)
    if after is None:
        turn(-ALIGN_TEST_DEG)
        return 0.0
    sign = 1.0 if _wrap90(after[0] - seen[0]) > 0 else -1.0
    delta = sign * _wrap90(want - after[0])
    turn(delta)
    return ALIGN_TEST_DEG + delta


def _centre_and_align(driver: ArmDriver, kin: Kinematics, cam, vcfg: VisionCfg) -> tuple[int, float]:
    steps, err = _centre(driver, kin, cam, vcfg)
    if _align_jaws(driver, kin, cam, vcfg):
        more, err = _centre(driver, kin, cam, vcfg)  # the camera sits off-axis, so turning shifts the view
        steps += more
    return steps, err


def find_and_grab(driver: ArmDriver, kin: Kinematics, cam, vcfg: VisionCfg, max_steps: int = 12) -> SeekResult:
    """Object already in view: point down, go to hover height, centre, grab."""
    table_z = _check_ready(driver, vcfg)
    if tilt_from_down(kin, driver.q_command()) > math.radians(2):
        point_down(driver, kin)
    _move_to_height(driver, kin, table_z + vcfg.hover_m)
    if locate(cam, vcfg) is None:
        raise GrabError("can't see the object from here; move the gripper over it (or re-click its colour)")
    steps, err = _centre_and_align(driver, kin, cam, vcfg)
    return SeekResult(grab=grab_below(driver, kin), steps=steps, final_error_px=err)


# ── fully automatic: get into position, search, centre, grab ─────────────

SEARCH_START_X = 0.26      # m in front of the base: middle of the pointing-down workspace
SEARCH_START_HEIGHT = 0.12  # m above the table before descending to hover height
TABLE_MARGIN = 0.02        # m every joint must stay above the table on big moves


def _down_rotation(kin: Kinematics) -> np.ndarray:
    # q2 - q3 - q4 = 90 deg points the gripper straight down (joints 2-4 pitch about parallel axes).
    return kin.fk(np.radians([0, 102, 100, -88, 0, 0]))[:3, :3]


def _check_clear_of_table(kin: Kinematics, q0: np.ndarray, q1: np.ndarray, table_z: float) -> None:
    """Joint moves don't follow straight lines; sample one and make sure nothing dips into the table."""
    for s in np.linspace(0.0, 1.0, 40):
        frames = kin.frames(q0 + s * (q1 - q0))
        low = min(T[2, 3] for T, axis in frames[2:])  # skip the base and shoulder, which sit on the table
        if low < table_z + TABLE_MARGIN:
            raise GrabError(f"the move to the search position would bring the arm within "
                            f"{(low - table_z) * 1000:.0f} mm of the table; raise the arm a little and retry")


def _move_to_down_pose(driver: ArmDriver, kin: Kinematics, xyz, table_z: float) -> None:
    q = driver.q_command()
    goal = np.eye(4)
    goal[:3, :3], goal[:3, 3] = _down_rotation(kin), xyz
    solutions = []
    for seed in down_seeds(q):
        try:
            solutions.append(kin.ik(goal, seed, iters=300))
        except IKError:
            pass
    if not solutions:
        raise GrabError(f"can't point down at ({xyz[0] * 1000:.0f}, {xyz[1] * 1000:.0f}, {xyz[2] * 1000:.0f}) mm")
    q_goal = min(solutions, key=lambda s: float(np.max(np.abs(s - q))))
    _check_clear_of_table(kin, q, q_goal, table_z)
    driver.move_to(q_goal, speed=0.5)


def _search_grid(vcfg: VisionCfg) -> list[tuple[float, float]]:
    """Serpentine sweep of the search area, spaced so neighbouring camera views overlap."""
    px_per_m = float(np.mean(np.linalg.norm(np.array(vcfg.jacobian_tool), axis=0)))
    w, h = vcfg.image_size or (640, 480)
    spacing = float(np.clip(0.6 * min(w, h) / px_per_m, 0.03, 0.10))
    xs = np.arange(vcfg.search_x_m[0], vcfg.search_x_m[1] + 1e-9, spacing)
    ys = np.arange(vcfg.search_y_m[0], vcfg.search_y_m[1] + 1e-9, spacing)
    cells = []
    for i, x in enumerate(xs):
        row = ys if i % 2 == 0 else ys[::-1]
        cells += [(float(x), float(y)) for y in row]
    return cells


def search_and_grab(driver: ArmDriver, kin: Kinematics, cam, vcfg: VisionCfg, status=lambda msg: None) -> SeekResult:
    """From wherever the arm is: get pointing down over the table, sweep until the object is
    in view, centre over it, grab it and lift."""
    table_z = _check_ready(driver, vcfg)
    if not driver.enabled:
        status("turning motors on")
        driver.enable()
    hover_z = table_z + vcfg.hover_m

    status("moving to the search position")
    _move_to_down_pose(driver, kin, [SEARCH_START_X, 0.0, table_z + max(SEARCH_START_HEIGHT, vcfg.hover_m)], table_z)
    _move_to_height(driver, kin, hover_z)

    found = locate(cam, vcfg) is not None
    if not found:
        cells = _search_grid(vcfg)
        for i, (x, y) in enumerate(cells, 1):
            status(f"searching {i}/{len(cells)}")
            here = kin.fk(driver.q_command())[:2, 3]
            try:
                _move_xy(driver, kin, np.array([x, y]) - here)
            except GrabError:
                continue  # cell out of reach from here; try the next one
            if locate(cam, vcfg) is not None:
                found = True
                break
    if not found:
        raise GrabError("searched the whole area and didn't see the object; check it's in front of the arm "
                        "and that the colour is learned (click it)")

    status("found it - centring")
    steps, err = _centre_and_align(driver, kin, cam, vcfg)
    status("grabbing")
    return SeekResult(grab=grab_below(driver, kin), steps=steps, final_error_px=err)


# ── look ahead, aim, approach, rotate down (camera looks along the gripper) ──

LOOK_ELBOW_DEG = (30, 30)           # q2, q3 of the look pose: tip ~10 cm up, ~25 cm out
LOOK_TILTS_DEG = (50, 65)           # gripper tilt below horizontal: far row, then near row
LOOK_YAWS_DEG = tuple(range(-60, 61, 20))
AIM_TEST_RAD = math.radians(3)
APPROACH_HEIGHT = 0.10              # m above the table where the tip stops before rotating down


def _camera_offset_tool(kin: Kinematics, vcfg: VisionCfg) -> np.ndarray:
    """Tool-frame vector from the tool point sideways to the camera's optical axis.

    From the pointing-down calibration: the object at `target_px` is under the tool point,
    and the image centre is where the optical axis meets the table, so their pixel gap
    divided by the image Jacobian is the offset. Assumes the camera looks parallel to the
    gripper, which the final pointing-down centring corrects for anyway.
    """
    if not vcfg.calibrated or vcfg.image_size is None:
        return np.zeros(3)
    J = np.array(vcfg.jacobian_tool)
    centre = np.array(vcfg.image_size, dtype=float) / 2
    a, b = -np.linalg.solve(J, centre - np.array(vcfg.target_px))  # tool-2D frame (see tool_yaw), m
    q_ref = np.radians([0, 102, 100, -88, 0, 0])                    # any pose; tool-frame vectors are constant
    R = kin.fk(q_ref)[:3, :3]
    approach = R.T @ kin.approach(q_ref)
    i = next(i for i in range(3) if abs(approach[i]) < 0.5)
    e = np.zeros(3)
    e[i] = 1.0
    return a * e + b * np.cross(-approach, e)


def sightline_hit(kin: Kinematics, q: np.ndarray, vcfg: VisionCfg, table_z: float) -> np.ndarray:
    """Where the camera's optical axis meets the table (base-frame x, y)."""
    T = kin.fk(q)
    origin = T[:3, 3] + T[:3, :3] @ _camera_offset_tool(kin, vcfg)
    d = kin.approach(q)
    if d[2] > -0.2:
        raise GrabError("the camera is looking too flat to judge distance; tilt it further down")
    t = (table_z - origin[2]) / d[2]
    return (origin + t * d)[:2]


def _image_centre(cam) -> np.ndarray:
    h, w = cam.fresh().shape[:2]
    return np.array([w / 2, h / 2])


def _safe_move(driver: ArmDriver, kin: Kinematics, q: np.ndarray, table_z: float, speed: float = 0.5) -> None:
    _check_clear_of_table(kin, driver.q_command(), q, table_z)
    driver.move_to(q, speed=speed)


def _look_for(driver: ArmDriver, kin: Kinematics, cam, vcfg: VisionCfg, table_z: float, status) -> bool:
    """Sweep the base left/right with the gripper (and camera) angled down at the table."""
    q2, q3 = LOOK_ELBOW_DEG
    for row, tilt in enumerate(LOOK_TILTS_DEG):
        yaws = LOOK_YAWS_DEG if row % 2 == 0 else LOOK_YAWS_DEG[::-1]
        for k, yaw in enumerate(yaws, 1):
            status(f"looking {row * len(yaws) + k}/{len(LOOK_TILTS_DEG) * len(yaws)}")
            q = np.radians([yaw, q2, q3, q2 - q3 - tilt, 0, 0])
            try:
                _safe_move(driver, kin, q, table_z)
            except GrabError:
                continue
            driver.sleep(0.2)
            if locate(cam, vcfg) is not None:
                return True
    return False


def _aim(driver: ArmDriver, kin: Kinematics, cam, vcfg: VisionCfg, table_z: float, max_steps: int = 10) -> None:
    """Turn the base (joint1) and tilt the wrist (joint4) until the object is in the image centre."""
    centre = _image_centre(cam)
    tol = 0.03 * centre[0] * 2
    p = locate(cam, vcfg)
    if p is None:
        raise GrabError("lost sight of the object")
    joints = (0, 3)
    J = np.zeros((2, 2))
    for col, j in enumerate(joints):  # measure how each joint moves the object in the image
        q0 = driver.q_command()
        q1 = q0.copy()
        q1[j] += AIM_TEST_RAD
        _safe_move(driver, kin, q1, table_z)
        p1 = locate(cam, vcfg)
        _safe_move(driver, kin, q0, table_z)
        if p1 is None:
            raise GrabError("lost the object during an aiming test move")
        J[:, col] = (p1 - p) / AIM_TEST_RAD
    for _ in range(max_steps):
        err = centre - p
        if np.linalg.norm(err) <= tol:
            return
        dq = GAIN * np.linalg.solve(J, err)
        dq = np.clip(dq, -0.2, 0.2)
        q = driver.q_command()
        q[0] += dq[0]
        q[3] += dq[1]
        _safe_move(driver, kin, q, table_z)
        driver.sleep(0.1)
        p = locate(cam, vcfg)
        if p is None:
            raise GrabError("lost sight of the object while aiming at it")
    if np.linalg.norm(centre - p) > 3 * tol:
        raise GrabError("couldn't aim the camera at the object")


def look_and_grab(driver: ArmDriver, kin: Kinematics, cam, vcfg: VisionCfg, status=lambda msg: None) -> SeekResult:
    """Spot the object with the gripper angled at the table, aim at it, move the tip above it,
    rotate to point straight down, then centre precisely and grab."""
    table_z = _check_ready(driver, vcfg)
    if not driver.enabled:
        status("turning motors on")
        driver.enable()

    if not _look_for(driver, kin, cam, vcfg, table_z, status):
        status("not seen from the side; searching from above")
        return search_and_grab(driver, kin, cam, vcfg, status)

    status("found it - aiming")
    _aim(driver, kin, cam, vcfg, table_z)
    xy = sightline_hit(kin, driver.q_command(), vcfg, table_z)
    status(f"it's at x={xy[0] * 1000:.0f} y={xy[1] * 1000:.0f} mm - approaching")

    # Tip straight to a point above the object, keeping the current angle...
    q = driver.q_command()
    tip = kin.fk(q)[:3, 3]
    above = np.array([xy[0], xy[1], table_z + max(APPROACH_HEIGHT, vcfg.hover_m)])
    try:
        driver.follow_path(kin.line(q, above - tip), duration=max(0.5, float(np.linalg.norm(above - tip)) / 0.08))
        # ...then rotate in place so it points straight down, tip still above the object.
        status("rotating to point down")
        point_down(driver, kin)
    except (IKError, GrabError):
        status("approaching with a joint move instead")
        _move_to_down_pose(driver, kin, above, table_z)
    _move_to_height(driver, kin, table_z + vcfg.hover_m)

    if locate(cam, vcfg) is None:  # a little off: small sweep around the estimate
        status("not in view from above; checking around it")
        saved = vcfg.search_x_m, vcfg.search_y_m
        vcfg.search_x_m, vcfg.search_y_m = [xy[0] - 0.04, xy[0] + 0.04], [xy[1] - 0.04, xy[1] + 0.04]
        try:
            for x, y in _search_grid(vcfg):
                here = kin.fk(driver.q_command())[:2, 3]
                try:
                    _move_xy(driver, kin, np.array([x, y]) - here)
                except GrabError:
                    continue
                if locate(cam, vcfg) is not None:
                    break
            else:
                raise GrabError("lost the object after approaching it")
        finally:
            vcfg.search_x_m, vcfg.search_y_m = saved

    status("centring")
    steps, err = _centre_and_align(driver, kin, cam, vcfg)
    status("grabbing")
    return SeekResult(grab=grab_below(driver, kin), steps=steps, final_error_px=err)
