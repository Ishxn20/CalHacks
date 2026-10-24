"""Arm driver: a background control loop that streams joint setpoints.

One thread owns the CAN bus and sends arm + gripper setpoints every tick, the
same pattern as Seeed's RebotArm control loop. Everything else (handoff
sequence, HTTP server, teach CLI) talks to the driver, never to the bus.

Motions are joint-space minimum-jerk trajectories between taught poses. All
joints start and stop together, so paths are smooth and repeatable.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import ArmConfig, PoseLibrary

log = logging.getLogger(__name__)


class ArmFault(RuntimeError):
    """The arm refused or stopped a motion; needs operator attention."""


class MotionAborted(RuntimeError):
    """A stop was requested while a motion was in progress."""


@dataclass
class MinJerk:
    q0: np.ndarray
    q1: np.ndarray
    t0: float
    duration: float

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, bool]:
        tau = (t - self.t0) / self.duration
        if tau >= 1.0:
            return self.q1.copy(), np.zeros_like(self.q1), True
        tau = max(tau, 0.0)
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        ds = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.duration
        dq = self.q1 - self.q0
        return self.q0 + s * dq, ds * dq, False


def min_jerk_duration(dq: np.ndarray, vmax: np.ndarray, min_s: float) -> float:
    # Peak velocity of a min-jerk profile is 1.875 * distance / duration.
    return max(float(np.max(1.875 * np.abs(dq) / vmax)), min_s)


class ArmDriver:
    def __init__(self, cfg: ArmConfig, bus, poses: PoseLibrary, *, sim: bool = False) -> None:
        self.cfg = cfg
        self.bus = bus
        self.poses = poses
        self.sim = sim
        n = cfg.n

        self._lock = threading.Lock()       # guards setpoints / measured state
        self._bus_lock = threading.Lock()   # serializes every bus call
        self._abort = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None

        self._enabled = False
        self._traj: MinJerk | None = None
        self._q_cmd = np.zeros(n)
        self._qd_cmd = np.zeros(n)
        self._g_cmd = (cfg.gripper.open, cfg.gripper.move_kp, cfg.gripper.move_kd)
        self._pos = np.full(n + 1, np.nan)
        self._tau = np.full(n + 1, np.nan)
        self._fault: str | None = None
        self._bus_errors = 0
        self._guard_from = 0.0              # collision guard ignores feedback before this time

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Connect, check every motor answers, and start the loop (motors stay disabled)."""
        self.bus.connect()
        offline = {k: v for k, v in self.bus.ping().items() if v}
        if offline:
            raise ArmFault(f"motors not responding: {offline}")
        with self._bus_lock:
            q = self.bus.read_fresh()
        with self._lock:
            self._pos = q
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="arm-control", daemon=True)
        self._thread.start()

    def shutdown(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._bus_lock:
            self.bus.close()
        self._enabled = False

    def enable(self) -> None:
        """Energize the motors, holding exactly where the arm is right now."""
        with self._bus_lock:
            q = self.bus.read_fresh()
        if np.isnan(q).any():
            raise ArmFault(f"could not read joint positions before enabling: {q}")
        with self._lock:
            self._traj = None
            self._q_cmd = q[:-1].copy()
            self._qd_cmd = np.zeros(self.cfg.n)
            self._g_cmd = (float(q[-1]), self.cfg.gripper.move_kp, self.cfg.gripper.move_kd)
            self._pos = q
        with self._bus_lock:
            self.bus.enable()
        with self._lock:
            # Cached feedback can still hold pre-limp positions for a few ticks.
            self._guard_from = time.perf_counter() + 0.5
            self._enabled = True

    def limp(self) -> None:
        """Cut torque so the arm can be moved by hand. It will fall under gravity, so support it."""
        with self._lock:
            self._enabled = False
            self._traj = None
        with self._bus_lock:
            self.bus.disable()

    # ── control loop ──────────────────────────────────────────────────────

    def _loop(self) -> None:
        period = 1.0 / self.cfg.control_rate_hz
        next_t = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            with self._lock:
                if self._traj is not None:
                    q, qd, done = self._traj.sample(now)
                    if done:
                        self._traj = None
                    self._q_cmd, self._qd_cmd = q, qd
                q, qd = self._q_cmd.copy(), self._qd_cmd.copy()
                g_pos, g_kp, g_kd = self._g_cmd
                enabled = self._enabled
            try:
                with self._bus_lock:
                    if enabled:
                        self.bus.send_arm(q, qd)
                        self.bus.send_gripper(g_pos, g_kp, g_kd)
                    self.bus.poll()
                    pos, tau = self.bus.cached()
                self._bus_errors = 0
            except Exception as e:  # noqa: BLE001 - any bus error counts toward the fault limit
                self._bus_errors += 1
                if self._bus_errors == 50:
                    self._set_fault(f"CAN bus errors: {e}")
                pos, tau = None, None
            if pos is not None:
                with self._lock:
                    self._pos, self._tau = pos, tau
                if enabled:
                    self._check_tracking(q, pos)

            next_t += period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    def _check_tracking(self, q_cmd: np.ndarray, pos: np.ndarray) -> None:
        limit = self.cfg.motion.tracking_error_limit
        if limit is None or self._fault or time.perf_counter() < self._guard_from:
            return
        err = np.abs(pos[:-1] - q_cmd)
        if np.nanmax(err, initial=0.0) > limit:
            i = int(np.nanargmax(err))
            self._set_fault(
                f"{self.cfg.joints[i].name} is {np.degrees(err[i]):.1f} deg off its setpoint "
                "(collision or blocked?) - arm frozen in place"
            )
            # Stop pushing: hold wherever the arm actually is.
            with self._lock:
                self._traj = None
                self._q_cmd = np.where(np.isnan(pos[:-1]), q_cmd, pos[:-1])
                self._qd_cmd = np.zeros(self.cfg.n)

    def _set_fault(self, msg: str) -> None:
        log.error("FAULT: %s", msg)
        self._fault = msg
        self._abort.set()

    # ── state ─────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def fault(self) -> str | None:
        return self._fault

    def clear_fault(self) -> None:
        self._fault = None
        self._abort.clear()

    def q_measured(self) -> np.ndarray:
        with self._lock:
            return self._pos[:-1].copy()

    def q_command(self) -> np.ndarray:
        with self._lock:
            return self._q_cmd.copy()

    def gripper_measured(self) -> float:
        with self._lock:
            return float(self._pos[-1])

    def read_fresh(self) -> np.ndarray:
        """Positions straight from the motors (works while limp). Length 7."""
        with self._bus_lock:
            return self.bus.read_fresh()

    def tracking_deviation(self) -> np.ndarray:
        """Measured minus commanded arm position (rad); NaN where unknown."""
        with self._lock:
            return self._pos[:-1] - self._q_cmd

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "moving": self._traj is not None,
                "fault": self._fault,
                "q_deg": _deg_list(self._pos[:-1]),
                "q_cmd_deg": _deg_list(self._q_cmd),
                "gripper_deg": _deg_list(self._pos[-1:])[0],
                "gripper_cmd_deg": round(float(np.degrees(self._g_cmd[0])), 2),
            }

    # ── motion ────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Freeze at the current setpoint and abort whatever motion is waiting."""
        self._abort.set()
        with self._lock:
            if self._traj is not None:
                q, _, _ = self._traj.sample(time.perf_counter())
                self._q_cmd = q
                self._traj = None
            self._qd_cmd = np.zeros(self.cfg.n)

    def clear_abort(self) -> None:
        if not self._fault:
            self._abort.clear()

    def _check_ready(self) -> None:
        if self._fault:
            raise ArmFault(self._fault)
        if self._abort.is_set():
            raise MotionAborted("stopped")
        if not self._enabled:
            raise ArmFault("motors are not enabled")

    def check_limits(self, q: np.ndarray) -> None:
        eps = 1e-6
        bad = [
            f"{j.name}={np.degrees(v):.1f} (limits {np.degrees(j.lower):.0f}..{np.degrees(j.upper):.0f})"
            for j, v in zip(self.cfg.joints, q)
            if not (j.lower - eps <= v <= j.upper + eps)
        ]
        if bad:
            raise ArmFault("target outside joint limits: " + ", ".join(bad))

    def move_to(self, q: np.ndarray, speed: float = 1.0, wait: bool = True) -> None:
        q = np.asarray(q, dtype=float)
        self._check_ready()
        self.check_limits(q)
        m = self.cfg.motion
        with self._lock:
            q0 = self._q_cmd.copy()
            vmax = self.cfg.max_vel * m.speed_scale * speed
            T = min_jerk_duration(q - q0, vmax, m.min_move_s)
            self._traj = MinJerk(q0=q0, q1=q.copy(), t0=time.perf_counter(), duration=T)
        if wait:
            self.wait_motion()

    def wait_motion(self) -> None:
        while True:
            with self._lock:
                moving = self._traj is not None
                target = self._traj.q1 if moving else self._q_cmd.copy()
            if not moving:
                break
            self.sleep(0.01)
        self._settle(target)

    def _settle(self, target: np.ndarray) -> None:
        m = self.cfg.motion
        deadline = time.perf_counter() + m.settle_timeout_s
        prev = self.q_measured()
        while time.perf_counter() < deadline:
            self.sleep(0.05)
            now = self.q_measured()
            if np.nanmax(np.abs(now - target), initial=0.0) < m.settle_tol:
                return
            # Stopped moving but short of target: gravity sag on a stiff-enough joint. Accept it.
            if np.nanmax(np.abs(now - prev), initial=0.0) < np.radians(0.2):
                return
            prev = now
        log.warning("move did not settle within %.1fs", m.settle_timeout_s)

    def sleep(self, s: float) -> None:
        if self._abort.wait(s):
            if self._fault:
                raise ArmFault(self._fault)
            raise MotionAborted("stopped")

    def pose_q(self, name: str) -> np.ndarray:
        pose = self.poses.get(name)
        if not pose.taught and not self.sim:
            raise ArmFault(f"pose {name!r} is a placeholder; teach it on the real arm first")
        return pose.q.copy()

    def move_pose(self, name: str, speed: float = 1.0, wrist_offset: float = 0.0) -> None:
        q = self.pose_q(name)
        q[-1] += wrist_offset
        self.move_to(q, speed)

    def move_path(self, names: list[str], speed: float = 1.0) -> None:
        for name in names:
            self.move_pose(name, speed)

    # ── gripper ───────────────────────────────────────────────────────────

    def _check_gripper_calibrated(self) -> None:
        if not self.cfg.gripper.calibrated and not self.sim:
            raise ArmFault("gripper open/closed angles are placeholders; run `gcal` in the teach tool")

    def set_gripper(self, pos: float, kp: float | None = None, kd: float | None = None) -> None:
        g = self.cfg.gripper
        if g.max is not None:
            pos = min(pos, g.max)   # past this the jaws hit their stop and the plastic breaks
        with self._lock:
            self._g_cmd = (float(pos), g.move_kp if kp is None else kp, g.move_kd if kd is None else kd)

    def gripper_open(self, wait_s: float = 0.4) -> None:
        self._check_gripper_calibrated()
        self.set_gripper(self.cfg.gripper.open)
        self.sleep(wait_s)

    def gripper_grasp(self) -> bool:
        """Squeeze softly toward fully closed. Returns True if the jaws stopped on an object."""
        self._check_gripper_calibrated()
        g = self.cfg.gripper
        self.set_gripper(g.closed, g.grip_kp, g.grip_kd)
        self.sleep(g.settle_s)
        # Read the motor directly: a stale cached value would look like "stopped short" = holding.
        pos = float(self.read_fresh()[-1])
        if np.isnan(pos):
            pos = self.gripper_measured()
        if np.isnan(pos):
            log.warning("no gripper feedback; assuming grasp succeeded")
            return True
        return abs(pos - g.closed) >= g.holding_min_gap


def _deg_list(a: np.ndarray) -> list[float | None]:
    return [None if np.isnan(x) else round(float(np.degrees(x)), 2) for x in a]
