"""Top-down grasping: grab whatever is directly below the gripper.

    grab_below(driver, kin)

1. Check the gripper is pointing (roughly) straight down.
2. Open the jaws.
3. Descend in a straight vertical line, keeping the gripper's orientation, until
   the fingertips are `clearance` above the table (if the table height is known),
   or until the arm feels contact (a joint deflects from its setpoint), whichever
   comes first. Without a table height it goes at most `max_depth` down.
4. After contact, come back up `backoff`.
5. Close with a constant squeeze torque; a stall short of fully closed = holding.
6. Lift straight up and check it's still in the jaws.

Tested with a rubber band lying flat: set the table height first (`table` in the
teach tool) so the fingertips finish ~1 mm off the surface.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from .driver import ArmDriver, ArmFault
from .kinematics import IKError, Kinematics, _axis_angle

log = logging.getLogger(__name__)

DOWN = np.array([0.0, 0.0, -1.0])
# Joints whose deflection shows the tool pushing into something below it (shoulder, elbow, wrist).
_CONTACT_JOINTS = [1, 2, 3, 4]


class GrabError(RuntimeError):
    pass


@dataclass
class GrabResult:
    holding: bool
    contact: bool              # stopped because it touched something
    start_z: float             # tool-point height before descending (m)
    grasp_z: float             # tool-point height when the jaws closed (m)
    jaw_gap_deg: float         # how far short of fully closed the jaws stopped
    note: str = ""

    def __str__(self) -> str:
        what = "HOLDING" if self.holding else "EMPTY"
        how = "touched down" if self.contact else "reached target depth"
        return (f"{what} - {how}, closed at z={self.grasp_z * 1000:.1f} mm "
                f"(descended {(self.start_z - self.grasp_z) * 1000:.0f} mm), jaw gap {self.jaw_gap_deg:.1f} deg"
                + (f"; {self.note}" if self.note else ""))


def tilt_from_down(kin: Kinematics, q: np.ndarray) -> float:
    """Angle (rad) between where the gripper points and straight down."""
    return math.acos(float(np.clip(np.dot(kin.approach(q), DOWN), -1.0, 1.0)))


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3)
    return _axis_angle(v / s, math.atan2(s, float(np.dot(a, b))))


def down_seeds(q: np.ndarray) -> list[np.ndarray]:
    """IK starting points that already point down. Joints 2-4 pitch about parallel axes, so the
    gripper points straight down when q2 - q3 - q4 = 90 deg; seed a spread of such elbow shapes."""
    seeds = [q]
    for q4 in (-85, -70, -55, -40):
        for q2 in (60, 80, 100, 120):
            q3 = q2 - 90 - q4
            if 0 <= q3 <= 179:
                s = q.copy()
                s[1:4] = np.radians([q2, q3, q4])
                seeds.append(s)
    return seeds


def point_down(driver: ArmDriver, kin: Kinematics, speed: float = 0.5) -> float:
    """Tilt the wrist so the gripper points straight down, keeping the tool point where it is.
    Returns the remaining tilt (rad).

    The B601-RS can only point straight down with the tool point within ~10-15 cm of the
    table (joint4 runs out of range higher up), so lower the arm first if this fails.
    """
    if not driver.enabled:
        raise ArmFault("motors are off; type `hold` first")
    q = driver.q_command()
    T = kin.fk(q)
    goal = T.copy()
    goal[:3, :3] = _rotation_between(kin.approach(q), DOWN) @ T[:3, :3]
    solutions = []
    for seed in down_seeds(q):
        try:
            solutions.append(kin.ik(goal, seed, iters=300))
        except IKError:
            pass
    if not solutions:
        raise GrabError(f"can't point straight down at z={T[2, 3] * 1000:.0f} mm; lower the gripper to "
                        "within ~10 cm of the table (and 10-40 cm out from the base) first")
    q_goal = min(solutions, key=lambda s: float(np.max(np.abs(s - q))))
    driver.move_to(q_goal, speed)
    return tilt_from_down(kin, q_goal)


def _line(kin: Kinematics, q: np.ndarray, dz: float) -> np.ndarray:
    try:
        return kin.line(q, np.array([0.0, 0.0, dz]))
    except IKError as e:
        raise GrabError(f"can't move straight {'down' if dz < 0 else 'up'} {abs(dz) * 1000:.0f} mm: {e}") from e


def _reachable_lift(kin: Kinematics, q: np.ndarray, lift: float) -> float:
    """Longest straight lift (lift, lift/2, lift/4) that stays reachable with the gripper pointing down."""
    for dz in (lift, lift / 2, lift / 4):
        try:
            kin.line(q, np.array([0.0, 0.0, dz]))
            return dz
        except IKError:
            pass
    raise GrabError(f"can't lift straight up from here; grab closer to the middle of the reach")


def grab_below(
    driver: ArmDriver,
    kin: Kinematics,
    *,
    table_z: float | None = None,
    depth: float | None = None,
    force: float | None = None,
) -> GrabResult:
    """Grab the object directly below the gripper. Distances in metres, force in Nm.

    table_z: tool-point height at the table surface (default: config grab.table_z_m).
    depth:   descend exactly this far instead (still stops early on contact).
    force:   squeeze torque (default: config gripper.grip_torque_nm).
    """
    cfg = driver.cfg.grab
    table_z = cfg.table_z if table_z is None else table_z
    if not driver.enabled:
        raise ArmFault("motors are not enabled")

    q0 = driver.q_command()
    tilt = tilt_from_down(kin, q0)
    if tilt > cfg.max_tilt:
        raise GrabError(f"gripper is {math.degrees(tilt):.0f} deg off straight down "
                        f"(max {math.degrees(cfg.max_tilt):.0f}); use point_down() first")
    z0 = float(kin.fk(q0)[2, 3])

    if depth is not None:
        z_goal = z0 - depth
    elif table_z is not None:
        z_goal = table_z + cfg.clearance
    else:
        z_goal = z0 - cfg.max_depth
    z_goal = max(z_goal, cfg.floor_z)                 # hard floor, even with no table height set
    if table_z is not None:
        z_goal = max(z_goal, table_z + cfg.clearance)  # never aim below the table
    if z_goal >= z0 - 0.001:
        raise GrabError(f"already at grasp height (z={z0 * 1000:.0f} mm, target {z_goal * 1000:.0f} mm)")

    # Plan the descent and the lift back out before moving, so an unreachable point fails
    # safely up here instead of with the object in the jaws at the bottom.
    down = _line(kin, q0, z_goal - z0)
    lift = _reachable_lift(kin, down[-1], cfg.lift)

    driver.gripper_open()

    # Gravity sag means joints never sit exactly on their setpoints; contact = change from rest.
    samples = []
    for _ in range(10):
        driver.sleep(0.02)
        samples.append(driver.tracking_deviation()[_CONTACT_JOINTS])
    baseline = np.nanmean(samples, axis=0)

    def touched() -> bool:
        dev = driver.tracking_deviation()[_CONTACT_JOINTS] - baseline
        return bool(np.nanmax(np.abs(dev), initial=0.0) > cfg.contact)

    log.info("descending %.0f mm", (z0 - z_goal) * 1000)
    completed = driver.follow_path(down, duration=(z0 - z_goal) / cfg.speed, stop_if=touched)
    contact = not completed
    if contact:
        # Arm is frozen with its setpoint slightly below where it actually stopped. Come back up
        # to `backoff` above the measured contact height so the jaws aren't scraping the surface.
        q_cmd = driver.q_command()
        z_touch = float(kin.fk(driver.q_measured())[2, 3])
        dz = (z_touch + cfg.backoff) - float(kin.fk(q_cmd)[2, 3])
        if abs(dz) > 0.0005:
            driver.follow_path(_line(kin, q_cmd, dz), duration=0.5)
        log.info("contact at z=%.1f mm", z_touch * 1000)

    grasp_z = float(kin.fk(driver.q_command())[2, 3])
    holding = driver.gripper_grasp(force)
    gap = math.degrees(abs(driver.gripper_measured() - driver.cfg.gripper.closed))

    q = driver.q_command()
    lift = _reachable_lift(kin, q, lift)
    driver.follow_path(_line(kin, q, lift), duration=lift / (2 * cfg.speed))

    note = ""
    if holding and not driver.gripper_holding():
        holding, note = False, "had it, but it slipped out while lifting"
    return GrabResult(holding=holding, contact=contact, start_z=z0, grasp_z=grasp_z, jaw_gap_deg=gap, note=note)
