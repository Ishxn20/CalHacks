"""The scrub-assistant task layer: pick -> verify -> present -> release.

One job runs at a time on a worker thread. The HTTP server (and anything
else) submits jobs and polls `status()`. `release()` and `stop()` are
signals that act on the running job immediately.

Handoff sequence for a request like "needle holder, 6-0 Prolene, backhand":

    hover over slot -> open -> descend -> grasp (check jaws closed on something)
    -> lift -> [tray-cam verify; on mismatch put it back and report]
    -> swing to handoff pose with wrist rotated for forehand/backhand
    -> wait for release (voice "release", API call, or a tug) -> open -> home
"""
from __future__ import annotations

import itertools
import logging
import math
import threading
import time
from collections import deque
from enum import Enum

import numpy as np

from .catalog import Catalog, Resolved
from .driver import ArmDriver, ArmFault, MotionAborted
from .verify import Verifier

log = logging.getLogger(__name__)


class Phase(str, Enum):
    IDLE = "idle"
    PICKING = "picking"
    VERIFYING = "verifying"
    RETURNING = "returning"
    MOVING = "moving"
    PRESENTING = "presenting"   # holding instrument out, waiting for the surgeon to take it
    RELEASING = "releasing"
    STAGED = "staged"           # holding the predicted next instrument near the handoff pose
    STOPPED = "stopped"
    FAULT = "fault"


class Busy(RuntimeError):
    pass


class TaskFailed(RuntimeError):
    """The job couldn't complete but the arm is in a safe, known state."""


class ScrubAssistant:
    def __init__(self, driver: ArmDriver, catalog: Catalog, verifier: Verifier, kin=None, reach=None) -> None:
        self.driver = driver
        self.catalog = catalog
        self.verifier = verifier
        self.cfg = driver.cfg
        self.kin = kin                      # Kinematics, only in pick.mode == "ik"
        self.reach = reach                  # sim.reach_policy.ReachPolicy, only with pick.reach == "rl"
        self._picked_from: tuple | None = None   # (x_mm, y_mm, yaw_deg, thickness_mm) of the last ik pick

        self.phase = Phase.IDLE
        self.holding: Resolved | None = None
        self._job: dict | None = None
        self._last_job: dict | None = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._release = threading.Event()
        self.events: deque[dict] = deque(maxlen=50)

    # ── public API (thread-safe) ──────────────────────────────────────────

    def submit(self, kind: str, resolved: Resolved | None = None) -> dict:
        fn = {
            "handoff": self._do_handoff,
            "stage": self._do_stage,
            "home": self._do_home,
            "return": self._do_return,
            "release": self._do_release,
        }[kind]
        with self._lock:
            if self._job is not None:
                raise Busy(f"busy with {self._job['kind']} #{self._job['id']}")
            job = {
                "id": next(self._ids), "kind": kind, "status": "running", "error": None,
                "request": resolved.describe() if resolved else None, "started": time.time(),
            }
            self._job = job
            self._release.clear()
        self.driver.clear_abort()
        threading.Thread(target=self._run, args=(job, fn, resolved), name=f"job-{job['id']}", daemon=True).start()
        return dict(job)

    def release(self) -> dict:
        """Surgeon has the instrument: open the jaws. Works mid-handoff or after a present timeout."""
        with self._lock:
            handing = self._job is not None and self._job["kind"] == "handoff"
        if handing:
            # Latched: if the arm is still on its way out, it releases as soon as it's presented.
            self._release.set()
            return {"status": "releasing" if self.phase == Phase.PRESENTING else "will release on arrival"}
        if self.holding is not None:
            return self.submit("release")
        return {"status": "nothing to release"}

    def reset(self) -> None:
        """Clear a fault or stop so new jobs are accepted. The arm stays where it is."""
        self.driver.clear_fault()
        if self.phase in (Phase.FAULT, Phase.STOPPED):
            self.phase = Phase.IDLE
        self._log("reset")

    def stop(self) -> None:
        self._log("stop requested")
        self.driver.stop()

    def status(self) -> dict:
        with self._lock:
            job = dict(self._job) if self._job else None
            last = dict(self._last_job) if self._last_job else None
        return {
            "phase": self.phase.value,
            "holding": self.holding.describe() if self.holding else None,
            "holding_slot": self.holding.slot.id if self.holding and self.holding.slot else None,
            "job": job,
            "last_job": last,
            "arm": self.driver.snapshot(),
            "events": list(self.events)[-10:],
        }

    def wait_idle(self, timeout: float = 60.0) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._job is None:
                    return dict(self._last_job) if self._last_job else None
            time.sleep(0.02)
        raise TimeoutError("job still running")

    # ── job runner ────────────────────────────────────────────────────────

    def _run(self, job: dict, fn, resolved: Resolved | None) -> None:
        try:
            msg = fn(resolved) if resolved is not None else fn()
            job.update(status="done", result=msg)
            self._log(f"{job['kind']} #{job['id']} done: {msg}")
        except MotionAborted:
            self.phase = Phase.STOPPED
            job.update(status="stopped", error="stopped by operator")
            self._log(f"{job['kind']} #{job['id']} stopped")
        except ArmFault as e:
            self.phase = Phase.FAULT
            job.update(status="fault", error=str(e))
            self._log(f"FAULT in {job['kind']} #{job['id']}: {e}")
        except TaskFailed as e:
            job.update(status="failed", error=str(e))
            self._log(f"{job['kind']} #{job['id']} failed: {e}")
        except Exception as e:  # noqa: BLE001 - never let the worker die silently
            log.exception("unexpected error in job")
            self.phase = Phase.FAULT
            job.update(status="fault", error=f"{type(e).__name__}: {e}")
        finally:
            job["finished"] = time.time()
            with self._lock:
                self._last_job = job
                self._job = None

    def _log(self, msg: str) -> None:
        log.info(msg)
        self.events.append({"t": round(time.time(), 2), "msg": msg})

    # ── jobs ──────────────────────────────────────────────────────────────

    def _do_handoff(self, r: Resolved) -> str:
        self._acquire(r)
        self._present(r)
        how = self._wait_for_take()
        if how == "timeout":
            return f"presented {r.describe()}; not taken, still holding"
        self._release_and_retreat()
        return f"handed {r.describe()} ({how})"

    def _do_stage(self, r: Resolved) -> str:
        self._acquire(r)
        self.phase = Phase.MOVING
        self.driver.move_to(self._oriented("staged", r.orientation))
        self.phase = Phase.STAGED
        return f"staged {r.describe()}"

    def _do_home(self) -> str:
        self.phase = Phase.MOVING
        self.driver.move_pose("home")
        self.phase = Phase.IDLE
        return "home" + (f" (holding {self.holding.describe()})" if self.holding else "")

    def _do_return(self) -> str:
        if self.holding is None:
            return "not holding anything"
        what = self.holding.describe()
        self._return_held()
        self.driver.move_pose("home")
        self.phase = Phase.IDLE
        return f"returned {what} to tray"

    def _do_release(self) -> str:
        what = self.holding.describe() if self.holding else "instrument"
        self._release_and_retreat()
        return f"released {what}"

    # ── steps ─────────────────────────────────────────────────────────────

    def _acquire(self, r: Resolved) -> None:
        """Make sure we're holding the requested instrument, picking it if needed."""
        if self.holding is not None:
            same = (self.holding.instrument.id == r.instrument.id) if r.slot is None else \
                (self.holding.slot is not None and self.holding.slot.id == r.slot.id)
            if same:
                self.holding = r          # same instrument; orientation may differ
                self._log(f"already holding {r.describe()}")
                return
            self._log(f"putting back {self.holding.describe()} first")
            self._return_held()
        if self.cfg.pick.mode == "ik":
            self._acquire_ik(r)
        else:
            self._pick(r)
        self._verify(r)

    def _acquire_ik(self, r: Resolved) -> None:
        """Look at the pile, uncover the requested instrument if it's buried, then pick it."""
        from .perception_client import PerceptionClient

        pk, d = self.cfg.pick, self.driver
        client = PerceptionClient(pk.perception_url)
        missed = False
        for _ in range(pk.max_declutter_moves):
            self.phase = Phase.MOVING
            d.move_pose("scan")
            d.sleep(0.3)               # wrist camera: let a frame from the settled scan pose arrive
            plan = client.plan(r.instrument.id)
            if plan["action"] == "fail":
                raise TaskFailed(plan.get("say") or f"I can't find the {r.instrument.name}")
            t = plan["target"]
            self._log(f"plan: {plan['action']} {t.get('id')} at ({t['x_mm']:.0f}, {t['y_mm']:.0f})")
            grasping = plan["action"] == "grasp_target"
            if not grasping:          # check the drop-off is reachable before lifting anything
                p = plan["place"]
                self._solve(p["x_mm"], p["y_mm"], pk.table_z + pk.hover, p["yaw_deg"])
                self._solve(p["x_mm"], p["y_mm"], pk.table_z + t["thickness_mm"] / 2000.0 + 0.005, p["yaw_deg"])
            try:
                # The requested instrument keeps its tip on a fixed side of the gripper so the taught
                # handoff pose presents it handle-first; movers can be pinched either way round.
                self._pick_at(t["x_mm"], t["y_mm"], t["yaw_deg"], t["thickness_mm"], allow_flip=not grasping)
            except TaskFailed:
                if missed:
                    raise
                missed = True        # the pile may have shifted: look again once
                continue
            if grasping:
                self.holding = r
                return
            self.holding = Resolved(slot=None, instrument=self._instrument(t.get("id")), orientation=None)
            self._log(f"carrying {self.holding.instrument.name} to staging")
            self._place_at(p["x_mm"], p["y_mm"], p["yaw_deg"], t["thickness_mm"])
        raise TaskFailed(f"couldn't uncover the {r.instrument.name} in {pk.max_declutter_moves} moves")

    def _pick(self, r: Resolved) -> None:
        d, slot = self.driver, r.slot
        slow = self.cfg.motion.approach_speed_scale
        self.phase = Phase.PICKING
        self._log(f"picking {r.describe()} from {slot.id}")
        d.move_pose(slot.approach[0])
        d.gripper_open()
        d.move_path(slot.approach[1:], speed=slow)
        got_it = d.gripper_grasp()
        if not got_it:
            d.gripper_open()
            d.move_path(list(reversed(slot.approach[:-1])), speed=slow)
            self.phase = Phase.IDLE
            raise TaskFailed(f"missed the grasp on {r.instrument.name} in {slot.id}; is it in the tray?")
        self.holding = r
        d.move_path(list(reversed(slot.approach[:-1])), speed=slow)

    def _verify(self, r: Resolved) -> None:
        if not self.verifier.enabled:
            return
        self.phase = Phase.VERIFYING
        verify_pose = self.driver.poses.poses.get("verify")
        if verify_pose is not None and (verify_pose.taught or self.driver.sim):
            self.driver.move_pose("verify")   # otherwise verify from the hover pose
        res = self.verifier.verify(r.instrument.id, r.slot.id if r.slot else None)
        self._log(f"tray-cam: ok={res.ok} detected={res.detected} conf={res.confidence} {res.note}".strip())
        if not res.ok:
            self._return_held()
            self.phase = Phase.IDLE
            raise TaskFailed(f"wrong instrument {'in ' + r.slot.id if r.slot else 'picked'}: {res.note}")

    def _return_held(self) -> None:
        assert self.holding is not None
        if self.holding.slot is None:      # ik pick: put it back where it came from, onto whatever is there now
            self._place_at(*self._picked_from, sense_contact=True)
            return
        d, slot = self.driver, self.holding.slot
        slow = self.cfg.motion.approach_speed_scale
        self.phase = Phase.RETURNING
        d.move_pose(slot.approach[0])
        d.move_path(slot.approach[1:], speed=slow)
        d.gripper_open()
        self.holding = None
        d.move_path(list(reversed(slot.approach[:-1])), speed=slow)

    # ── ik picking (pick.mode == "ik") ────────────────────────────────────

    def _solve(self, x_mm: float, y_mm: float, z_m: float, yaw_deg: float, allow_flip: bool = True) -> np.ndarray:
        """Joint vector for a top-down grasp pose, or TaskFailed if the arm can't reach it."""
        assert self.kin is not None, "pick.mode is not 'ik'"
        if self.reach is not None:
            return self._solve_rl(x_mm, y_mm, z_m, yaw_deg, allow_flip)
        q = self.kin.ik_topdown(x_mm / 1000.0, y_mm / 1000.0, z_m, math.radians(yaw_deg),
                                self.driver.q_command(), (self.cfg.lower, self.cfg.upper), allow_flip=allow_flip)
        if q is None:
            raise TaskFailed(f"can't reach ({x_mm:.0f}, {y_mm:.0f}) mm at {z_m * 1000:.0f} mm")
        return q

    def _solve_rl(self, x_mm: float, y_mm: float, z_m: float, yaw_deg: float, allow_flip: bool) -> np.ndarray:
        """The reach policy's joint vector, accepted only if it is within limits and FK puts the tool within
        3 mm and 3 deg of the pose. No fallback to IK: a refusal is a TaskFailed, reported like any other."""
        p = np.array([x_mm / 1000.0, y_mm / 1000.0, z_m])
        q = self.reach.solve(*p, math.radians(yaw_deg), self.driver.q_command(), allow_flip=allow_flip)
        where = f"({x_mm:.0f}, {y_mm:.0f}) mm at {z_m * 1000:.0f} mm"
        if q is None:
            raise TaskFailed(f"reach policy didn't converge on {where}")
        try:
            self.driver.check_limits(q)
        except ArmFault as e:
            raise TaskFailed(f"reach policy for {where}: {e}") from e
        pos, R = self.kin.fk(q)
        d_mm = float(np.linalg.norm(pos - p)) * 1000
        yaws = (yaw_deg, yaw_deg + 180.0) if allow_flip else (yaw_deg,)
        rot_deg = min(math.degrees(math.acos(np.clip((np.trace(self.kin._target_rotation(math.radians(y)).T @ R) - 1)
                                                     / 2, -1, 1))) for y in yaws)
        if d_mm > 3.0 or rot_deg > 3.0:
            raise TaskFailed(f"reach policy for {where} is off by {d_mm:.1f} mm / {rot_deg:.1f} deg")
        return q

    def _descend_until_contact(self, x_mm: float, y_mm: float, yaw_deg: float, z_floor: float,
                               allow_flip: bool = True) -> float:
        """Step the open gripper straight down from hover until the pile pushes back or z_floor is reached.

        Contact = any pick.contact_joints lagging its setpoint by more than pick.contact_deviation beyond
        the lag measured at hover (gravity sag). On contact, back off 1 mm and return that z (m).
        Contact on the very first step means something is in the way at hover height: TaskFailed.
        """
        pk, d = self.cfg.pick, self.driver
        idx = [self.cfg.joint_index(n) for n in pk.contact_joints]
        slow = self.cfg.motion.approach_speed_scale
        z_hover = pk.table_z + pk.hover
        solve = lambda z: self._solve(x_mm, y_mm, z, yaw_deg, allow_flip)  # noqa: E731
        d.move_to(solve(z_hover))
        samples = []
        for _ in range(5):
            d.sleep(0.02)
            samples.append(d.tracking_deviation()[idx])
        baseline = np.nanmean(samples, axis=0)
        z = z_hover
        while z > z_floor + 1e-9:
            z = max(z - pk.descend_step, z_floor)
            d.move_to(solve(z), speed=slow)
            d.sleep(0.02)   # let feedback catch up before judging contact
            dev = np.nanmax(np.abs(d.tracking_deviation()[idx] - baseline), initial=0.0)
            if dev > pk.contact_deviation:
                if z >= z_hover - pk.descend_step - 1e-9:
                    d.move_to(solve(z_hover), speed=slow)
                    raise TaskFailed("something is in the way above the pile")
                z = min(z + 0.001, z_hover)
                d.move_to(solve(z), speed=slow)
                return z
        return z

    def _pick_at(self, x_mm: float, y_mm: float, yaw_deg: float, thickness_mm: float,
                 allow_flip: bool = True) -> None:
        """hover -> open -> contact descent -> grasp -> lift. TaskFailed on a miss (caller decides on retry).
        allow_flip=False keeps the instrument's tip on a fixed side of the gripper (see Kinematics.ik_topdown)."""
        pk, d = self.cfg.pick, self.driver
        z_hover = pk.table_z + pk.hover
        q_hover = self._solve(x_mm, y_mm, z_hover, yaw_deg, allow_flip)   # fail before moving if unreachable
        self.phase = Phase.PICKING
        d.move_to(q_hover)
        d.gripper_open()
        z_floor = pk.table_z + thickness_mm / 2000.0 - 0.002
        self._descend_until_contact(x_mm, y_mm, yaw_deg, z_floor, allow_flip)
        got_it = d.gripper_grasp()
        d.move_to(q_hover, speed=self.cfg.motion.approach_speed_scale)
        if not got_it:
            d.gripper_open()
            raise TaskFailed(f"missed the grasp at ({x_mm:.0f}, {y_mm:.0f}) mm")
        self._picked_from = (x_mm, y_mm, yaw_deg, thickness_mm)

    def _place_at(self, x_mm: float, y_mm: float, yaw_deg: float, thickness_mm: float,
                  sense_contact: bool = False) -> None:
        """hover -> descend -> open -> hover. sense_contact: stop on whatever is already there (putting an
        instrument back onto the pile) instead of going down to just above the table."""
        pk, d = self.cfg.pick, self.driver
        slow = self.cfg.motion.approach_speed_scale
        z_place = pk.table_z + thickness_mm / 2000.0 + 0.005
        q_hover = self._solve(x_mm, y_mm, pk.table_z + pk.hover, yaw_deg)
        q_down = self._solve(x_mm, y_mm, z_place, yaw_deg)       # both solved before any motion
        self.phase = Phase.RETURNING
        d.move_to(q_hover)
        if sense_contact:
            try:
                self._descend_until_contact(x_mm, y_mm, yaw_deg, z_place)
            except TaskFailed:
                pass              # something right under the hover point: let go from there, gently held above
        else:
            d.move_to(q_down, speed=slow)
        d.gripper_open()
        self.holding = None
        d.move_to(q_hover, speed=slow)

    def _instrument(self, fid: str | None):
        from .config import Instrument

        if fid and fid in self.catalog.tray.instruments:
            return self.catalog.tray.instruments[fid]
        return Instrument(id=fid or "unknown", name=(fid or "instrument").replace("_", " "), category="unknown",
                          orientable=False, aliases=[])

    def _oriented(self, pose: str, orientation: str | None) -> np.ndarray:
        """Pose with the wrist (joint6) rotated for forehand/backhand, wrapped into joint limits."""
        q = self.driver.pose_q(pose)
        if orientation is None:
            return q
        offsets = self.cfg.handoff.orientation_offsets
        if orientation not in offsets:
            raise TaskFailed(f"no wrist offset configured for {orientation!r}")
        j6 = self.cfg.joints[-1]
        base = q[-1]
        options = [base + offsets[orientation] + k * 2 * math.pi for k in (0, -1, 1)]
        valid = [v for v in options if j6.lower <= v <= j6.upper]
        if not valid:
            raise ArmFault(
                f"{orientation} needs joint6 at {math.degrees(options[0]):.0f} deg, outside limits; "
                f"re-teach '{pose}' with the wrist nearer the middle of its range"
            )
        q[-1] = min(valid, key=lambda v: abs(v - base))
        return q

    def _present(self, r: Resolved) -> None:
        self.phase = Phase.MOVING
        self._log(f"presenting {r.describe()}")
        self.driver.move_to(self._oriented("handoff", r.orientation))
        self.phase = Phase.PRESENTING

    def _wait_for_take(self) -> str:
        h = self.cfg.handoff
        tug = h.tug
        idx = [self.cfg.joint_index(n) for n in tug.joints] if tug.enabled else []
        baseline = None
        if idx:
            # Gravity sag means the wrist never sits exactly on its setpoint; tug = change from rest.
            samples = []
            for _ in range(15):
                self.driver.sleep(0.02)
                samples.append(self.driver.tracking_deviation()[idx])
            baseline = np.nanmean(samples, axis=0)
        deadline = time.monotonic() + h.present_timeout_s
        tug_since = None
        while time.monotonic() < deadline:
            if self._release.is_set():
                return "release command"
            self.driver.sleep(0.01)
            if baseline is not None:
                dev = np.nanmax(np.abs(self.driver.tracking_deviation()[idx] - baseline), initial=0.0)
                if dev > tug.deviation:
                    tug_since = tug_since or time.monotonic()
                    if time.monotonic() - tug_since >= tug.hold_s:
                        return "tug"
                else:
                    tug_since = None
        self._log("present timed out; still holding")
        return "timeout"

    def _release_and_retreat(self) -> None:
        self.phase = Phase.RELEASING
        self.driver.gripper_open(wait_s=self.cfg.handoff.release_wait_s)
        self.holding = None
        self.phase = Phase.MOVING
        self.driver.move_pose("home")
        self.phase = Phase.IDLE
