"""Motor bus backends.

`RobstrideBus` talks to the real B601-RS over CAN via motorbridge.
`SimBus` is a kinematic stand-in with the same interface so the whole stack
(server, handoff sequence, teach tool) runs on a laptop with no arm attached.

Arrays are length 7: joint1..joint6 then the gripper motor.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from .config import ArmConfig

log = logging.getLogger(__name__)

# RobStride parameter IDs (same ones Seeed's reBotArm_control_py uses).
RS_MECH_POS = 0x7019
RS_LIMIT_SPD = 0x7017
RS_SPD_KP = 0x701F
RS_SPD_KI = 0x7020
RS_LOC_KP = 0x701E


class BusError(RuntimeError):
    pass


class RobstrideBus:
    """All seven RobStride motors on one CAN channel."""

    def __init__(self, cfg: ArmConfig) -> None:
        self.cfg = cfg
        self._ctrl = None
        self._motors: list = []
        self._arm_mode = cfg.arm_mode

    def connect(self) -> None:
        try:
            from motorbridge import Controller
        except ImportError as e:
            raise BusError(
                "motorbridge is not installed (pip install motorbridge). It has no wheel for "
                "Intel macOS; use Ubuntu or an Apple Silicon Mac, or run with --sim."
            ) from e
        self._ctrl = Controller(self.cfg.channel)
        g = self.cfg.gripper
        specs = [(j.motor_id, j.feedback_id, j.model) for j in self.cfg.joints]
        specs.append((g.motor_id, g.feedback_id, g.model))
        self._motors = [self._ctrl.add_robstride_motor(mid, fid, model) for mid, fid, model in specs]
        if self.cfg.active_report:
            for name, m in zip(self.names(), self._motors):
                try:
                    m.robstride_set_active_report(True)
                except Exception as e:  # noqa: BLE001 - not fatal; feedback may just be staler
                    log.warning("could not enable active report on %s: %s", name, e)

    def names(self) -> list[str]:
        return [j.name for j in self.cfg.joints] + ["gripper"]

    def ping(self) -> dict[str, str | None]:
        """Return {motor name: None if online else error text}."""
        out: dict[str, str | None] = {}
        for name, m in zip(self.names(), self._motors):
            try:
                m.robstride_ping()
                out[name] = None
            except Exception as e:  # noqa: BLE001 - surface any driver error verbatim
                out[name] = str(e)
        return out

    def enable(self) -> None:
        from motorbridge import Mode

        # RobStride only accepts run-mode changes while disabled, so set modes first.
        for j, m in zip(self.cfg.joints, self._motors[:6]):
            if self._arm_mode == "pos_vel":
                for pid, val in ((RS_LIMIT_SPD, j.pv_vlim), (RS_SPD_KP, j.pv_vel_kp),
                                 (RS_SPD_KI, j.pv_vel_ki), (RS_LOC_KP, j.pv_pos_kp)):
                    m.robstride_write_param_f32(pid, val)
                    time.sleep(0.01)
                m.ensure_mode(Mode.POS_VEL, 1000)
            else:
                m.ensure_mode(Mode.MIT, 1000)
            time.sleep(0.02)
        self._motors[6].ensure_mode(Mode.MIT, 1000)
        time.sleep(0.1)
        self._ctrl.enable_all()
        time.sleep(0.05)

    def disable(self) -> None:
        if self._ctrl is not None:
            self._ctrl.disable_all()

    def send_arm(self, q: np.ndarray, qd: np.ndarray) -> None:
        for j, m, qi, qdi in zip(self.cfg.joints, self._motors[:6], q, qd):
            if self._arm_mode == "pos_vel":
                m.send_pos_vel(float(qi), float(j.pv_vlim))
            else:
                m.send_mit(float(qi), float(qdi), j.kp, j.kd, 0.0)

    def send_gripper(self, pos: float, kp: float, kd: float) -> None:
        self._motors[6].send_mit(float(pos), 0.0, float(kp), float(kd), 0.0)

    def poll(self) -> None:
        self._ctrl.poll_feedback_once()

    def cached(self) -> tuple[np.ndarray, np.ndarray]:
        """Latest (position, torque) from feedback frames; NaN where nothing has arrived."""
        pos = np.full(7, np.nan)
        tau = np.full(7, np.nan)
        for i, m in enumerate(self._motors):
            st = m.get_state()
            if st is not None:
                pos[i], tau[i] = st.pos, st.torq
        return pos, tau

    def read_fresh(self) -> np.ndarray:
        """Read mechanical position straight from each motor's parameter table.

        RobStride has no on-demand status request, and cached state goes stale
        while motors are disabled, so this is the reliable read when limp.
        """
        out = np.full(7, np.nan)
        for i, m in enumerate(self._motors):
            try:
                out[i] = m.robstride_get_param_f32(RS_MECH_POS, 200)
            except Exception:  # noqa: BLE001 - caller checks for NaN
                pass
        return out

    def close(self) -> None:
        if self._ctrl is None:
            return
        try:
            self._ctrl.disable_all()
        finally:
            for m in self._motors:
                m.close()
            self._ctrl.shutdown()
            self._ctrl.close()
            self._ctrl = None
            self._motors = []


class SimBus:
    """Kinematic simulator: joints chase their setpoints with a velocity cap.

    Knobs for tests/demos:
      object_in_jaws  - gripper stalls short of closed (successful grasp) when True
      external_offset - added to measured positions (e.g. simulate a surgeon's tug)
      sag             - where each joint settles relative to its setpoint (gravity on the real arm's stiff-ish
                        MIT joints: joint 2 sat 4 deg low, joint 3 2 deg short)
    """

    def __init__(self, cfg: ArmConfig, start: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.pos = np.zeros(7) if start is None else np.array(start, dtype=float)
        self.target = self.pos.copy()
        self.enabled = False
        self.object_in_jaws = True
        self.external_offset = np.zeros(7)
        self.sag = np.zeros(7)
        self._last = time.perf_counter()
        self._vmax = np.append(cfg.max_vel * 100.0, math.radians(2000))

    def connect(self) -> None:
        pass

    def names(self) -> list[str]:
        return [j.name for j in self.cfg.joints] + ["gripper"]

    def ping(self) -> dict[str, str | None]:
        return {n: None for n in self.names()}

    def enable(self) -> None:
        self.enabled = True
        self.target = self.pos.copy()

    def disable(self) -> None:
        self.enabled = False

    def send_arm(self, q: np.ndarray, qd: np.ndarray) -> None:
        self.target[:6] = q

    def send_gripper(self, pos: float, kp: float, kd: float) -> None:
        g = self.cfg.gripper
        if self.object_in_jaws:
            # Jaws stop on the instrument shaft partway between open and closed.
            stall = g.open + 0.6 * (g.closed - g.open)
            lo, hi = sorted((g.open, stall))
            pos = min(max(pos, lo), hi)
        self.target[6] = pos

    def poll(self) -> None:
        now = time.perf_counter()
        dt, self._last = now - self._last, now
        if self.enabled:
            step = np.clip(self.target + self.sag - self.pos, -self._vmax * dt, self._vmax * dt)
            self.pos = self.pos + step

    def cached(self) -> tuple[np.ndarray, np.ndarray]:
        return self.pos + self.external_offset, np.zeros(7)

    def read_fresh(self) -> np.ndarray:
        return self.pos + self.external_offset

    def close(self) -> None:
        self.enabled = False
