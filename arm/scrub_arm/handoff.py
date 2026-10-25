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
JOINT_ROOM_DEG = 10.0   # hover must leave every joint this far from its limit, or come in from the other side


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
    def __init__(self, driver: ArmDriver, catalog: Catalog, verifier: Verifier, kin=None) -> None:
        self.driver = driver
        self.catalog = catalog
        self.verifier = verifier
        self.cfg = driver.cfg
        self.kin = kin                      # Kinematics, only in pick.mode == "ik"
        self._picked_from: tuple | None = None   # (x_mm, y_mm, yaw_deg, thickness_mm) of the last ik pick
        self._flipped = False               # last pick's jaws 180 deg from the plan's yaw: present with the wrist turned
        # Optional closed-loop alignment at hover (pick.reach rl): (x_mm, y_mm, yaw_deg, thickness_mm) ->
        # corrected (x_mm, y_mm, yaw_deg) to descend at. Set by the launcher that owns the camera.
        self.refine = None

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
        self.driver.move_to(self._oriented("staged", r.orientation, self._flipped and self.kin is not None))
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
            d.gripper_open()           # closed fingers hang into the middle of the wrist camera's view
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
                # Either way round: the pinch is symmetric, and a tilted gripper can only lean away from the
                # base (yaw 48 was unreachable at (445, 18), yaw -132 fine). A flipped requested instrument is
                # presented with the wrist turned 180 deg (_flipped), still handle-first.
                self._pick_at(t["x_mm"], t["y_mm"], t["yaw_deg"], t["thickness_mm"], fid=t.get("id"))
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

    def _solve(self, x_mm: float, y_mm: float, z_m: float, yaw_deg: float, allow_flip: bool = True,
               tilt: bool | None = None) -> np.ndarray:
        """Joint vector for a grasp pose (the pick tilt where it reaches, else top-down; see Kinematics.ik_topdown),
        or TaskFailed if the arm can't reach it."""
        assert self.kin is not None, "pick.mode is not 'ik'"
        q = self.kin.ik_topdown(x_mm / 1000.0, y_mm / 1000.0, z_m, math.radians(yaw_deg),
                                self.driver.q_command(), (self.cfg.lower, self.cfg.upper), allow_flip=allow_flip,
                                tilt=tilt)
        if q is None:
            raise TaskFailed(f"can't reach ({x_mm:.0f}, {y_mm:.0f}) mm at {z_m * 1000:.0f} mm")
        return q

    def _pick_side(self, x_mm: float, y_mm: float, yaw_deg: float, z_lo: float, z_hi: float,
                   allow_flip: bool = True) -> tuple[float, bool]:
        """Which way round to come in: the planned jaw yaw, or from the other side (yaw + 180; the pinch is
        symmetric) when the planned way leaves a joint within JOINT_ROOM_DEG of its limit at the hover. Real run
        20261024-194649: joint 4 at -88.5 of -89 deg at x 495, so the centring's 1 mm / 3 deg corrections were
        unsolvable. Neither side with that room: the one with the most. Returns (yaw_deg, tilt); TaskFailed if
        neither side reaches (fails before anything moves)."""
        best = None
        for yaw in (yaw_deg, yaw_deg + 180.0) if allow_flip else (yaw_deg,):
            tilt = self._tilt_for(x_mm, y_mm, yaw, z_lo, z_hi, allow_flip=False)
            try:
                q = self._solve(x_mm, y_mm, z_hi, yaw, False, tilt)
                self._solve(x_mm, y_mm, z_lo, yaw, False, tilt)
            except TaskFailed:
                continue
            room = float(np.degrees(np.minimum(q - self.cfg.lower, self.cfg.upper - q).min()))
            if room >= JOINT_ROOM_DEG:
                return yaw, tilt
            if best is None or room > best[0]:
                best = (room, yaw, tilt)
        if best is None:
            raise TaskFailed(f"can't reach ({x_mm:.0f}, {y_mm:.0f}) mm at {z_hi * 1000:.0f} mm")
        self._log(f"tight reach at ({x_mm:.0f}, {y_mm:.0f}): a joint is {best[0]:.0f} deg from its limit either way")
        return best[1], best[2]

    def _tilt_for(self, x_mm: float, y_mm: float, yaw_deg: float, z_lo: float, z_hi: float,
                  allow_flip: bool = True) -> bool:
        """One orientation for a whole hover -> table -> lift: the pick tilt if it reaches both ends here, else
        straight down (solving each height on its own mixed the two mid-descent, and the lift couldn't follow)."""
        if self.kin.tilt_R is None:
            return False
        try:
            self._solve(x_mm, y_mm, z_hi, yaw_deg, allow_flip, tilt=True)
            self._solve(x_mm, y_mm, z_lo, yaw_deg, allow_flip, tilt=True)
            return True
        except TaskFailed:
            return False

    def move_line(self, dx_mm: float, dy_mm: float, dz_mm: float, step_mm: float = 2.0, speed: float = 0.3,
                  dyaw_deg: float = 0.0, step_deg: float = 3.0, exact: bool = True) -> None:
        """Move the gripper in a straight line by (dx, dy, dz) mm in the table frame (x out from the base, y left,
        z up), holding its orientation (or turning it by dyaw_deg about the vertical through the fingertips, e.g. to
        square the jaws at the hover): re-solved every step_mm / step_deg so the path stays straight, instead of one joint-space
        move (which dipped and swung the gripper on the real arm). The current gravity-sag offset is carried along;
        the end point is reached exactly (exact=False: no correction at the end, e.g. lowering onto the mat, where
        correcting would push down harder)."""
        d = self.driver
        q0 = d.q_measured()
        off = d.q_command() - q0                                # the setpoint offset holding it here (sag)
        p0, R0 = self.kin.fk(q0)
        delta = np.array([dx_mm, dy_mm, dz_mm]) / 1000.0
        n = max(1, int(np.ceil(max(np.linalg.norm(delta) * 1000.0 / step_mm, abs(dyaw_deg) / step_deg))))
        q, qs = q0, []
        for i in range(1, n + 1):                      # solve the whole line first: nothing moves if any of it fails
            t = np.radians(dyaw_deg) * i / n
            Rz = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
            q = self.kin.ik_pose(p0 + delta * i / n, Rz @ R0, q, (self.cfg.lower, self.cfg.upper))
            if q is None:
                raise TaskFailed(f"can't move straight by ({dx_mm:.0f}, {dy_mm:.0f}, {dz_mm:.0f}) mm from here")
            qs.append(q)
        d.move_through([qq + off for qq in qs], speed=speed)  # one smooth motion (a stop every 2 mm took ~20 s per 10 cm)
        if exact:
            d.move_to(q, speed=speed, exact=True)

    def _descend_until_contact(self, x_mm: float, y_mm: float, yaw_deg: float, z_floor: float,
                               allow_flip: bool = True, tilt: bool | None = None) -> float:
        """Step the open gripper straight down from hover until the pile pushes back or z_floor is reached.

        Contact = any pick.contact_joints lagging its setpoint by more than pick.contact_deviation beyond
        the lag measured at hover (gravity sag). On contact, back off 1 mm and return that z (m).
        Contact on the very first step means something is in the way at hover height: TaskFailed.
        """
        pk, d = self.cfg.pick, self.driver
        idx = [self.cfg.joint_index(n) for n in pk.contact_joints]
        slow = self.cfg.motion.approach_speed_scale
        z_hover = pk.table_z + pk.hover
        # The hover is an IK target: correct the arm's gravity sag there (it sat ~3 cm low on the real arm), then
        # carry that same setpoint offset down every step. Correcting each step instead would also "correct"
        # against the pile pushing back -- pressing harder exactly when contact should stop it.
        off = d.move_to(self._solve(x_mm, y_mm, z_hover, yaw_deg, allow_flip, tilt), exact=True)
        solve = lambda z: self._solve(x_mm, y_mm, z, yaw_deg, allow_flip, tilt) + off  # noqa: E731
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
                 allow_flip: bool = True, fid: str | None = None) -> None:
        """hover -> open -> centre (refine) -> straight down to grasp height -> grasp -> straight up. TaskFailed on
        a miss (caller decides on retry).
        allow_flip=False keeps the instrument's tip on a fixed side of the gripper (see Kinematics.ik_topdown)."""
        pk, d = self.cfg.pick, self.driver
        plan_yaw = yaw_deg
        z_hover = pk.table_z + pk.hover
        z_grasp = pk.table_z + thickness_mm / 2000.0              # jaws close on the instrument's mid-height
        yaw_deg, tilt = self._pick_side(x_mm, y_mm, yaw_deg, z_grasp, z_hover, allow_flip)
        q_hover = self._solve(x_mm, y_mm, z_hover, yaw_deg, False, tilt)
        self.phase = Phase.PICKING
        d.move_to(q_hover, exact=True)            # an IK target: cancel gravity sag (the RL alignment starts here)
        d.gripper_open()
        if self.refine is not None:      # centres it at the hover: (x, y, yaw[, z]) where it left the jaws
            # fid: which instrument to centre on (tracking "whatever is biggest" chased glare on the real arm)
            x_mm, y_mm, yaw_deg, *_ = self.refine(x_mm, y_mm, yaw_deg, thickness_mm, **({"fid": fid} if fid else {}))
        # Ready: one smooth straight-down move to grasp height from wherever it was centred -- no adjusting, no
        # steps, no contact checks on the way (user); then close and go straight back up.
        slow = self.cfg.motion.approach_speed_scale
        z_now = self.kin.fk(d.q_measured())[0][2]
        self.move_line(0, 0, (z_grasp - z_now) * 1000, speed=slow, exact=False)
        got_it = d.gripper_grasp()
        # straight up (a joint-space move back to the hover pose can swing the gripper through the pile)
        self.move_line(0, 0, (z_hover - z_grasp) * 1000, speed=slow)
        if not got_it:
            d.gripper_open()
            raise TaskFailed(f"missed the grasp at ({x_mm:.0f}, {y_mm:.0f}) mm")
        self._picked_from = (x_mm, y_mm, yaw_deg, thickness_mm)
        _, R = self.kin.fk(d.q_measured())
        jaw = R @ self.kin.jaw_axis_local
        self._flipped = abs((math.degrees(math.atan2(jaw[1], jaw[0])) - plan_yaw + 180.0) % 360.0 - 180.0) > 90.0

    def _place_at(self, x_mm: float, y_mm: float, yaw_deg: float, thickness_mm: float,
                  sense_contact: bool = False) -> None:
        """hover -> descend -> open -> hover. sense_contact: stop on whatever is already there (putting an
        instrument back onto the pile) instead of going down to just above the table."""
        pk, d = self.cfg.pick, self.driver
        slow = self.cfg.motion.approach_speed_scale
        z_place = pk.table_z + thickness_mm / 2000.0 + 0.005
        tilt = self._tilt_for(x_mm, y_mm, yaw_deg, z_place, pk.table_z + pk.hover)
        q_hover = self._solve(x_mm, y_mm, pk.table_z + pk.hover, yaw_deg, tilt=tilt)
        q_down = self._solve(x_mm, y_mm, z_place, yaw_deg, tilt=tilt)       # both solved before any motion
        self.phase = Phase.RETURNING
        off = d.move_to(q_hover, exact=True)      # sag-corrected: uncorrected, the real arm sat ~3 cm low, which
        if sense_contact:                         # would have put this "5 mm above the table" into the table
            try:
                self._descend_until_contact(x_mm, y_mm, yaw_deg, z_place, tilt=tilt)
            except TaskFailed:
                pass              # something right under the hover point: let go from there, gently held above
        else:
            d.move_to(q_down + off, speed=slow)
        d.gripper_open()
        self.holding = None
        d.move_to(q_hover, speed=slow)

    def _instrument(self, fid: str | None):
        from .config import Instrument

        if fid and fid in self.catalog.tray.instruments:
            return self.catalog.tray.instruments[fid]
        return Instrument(id=fid or "unknown", name=(fid or "instrument").replace("_", " "), category="unknown",
                          orientable=False, aliases=[])

    def _oriented(self, pose: str, orientation: str | None, flipped: bool = False) -> np.ndarray:
        """Pose with the wrist (joint6) rotated for forehand/backhand, plus 180 deg if the instrument was picked
        the other way round (flipped), wrapped into joint limits."""
        q = self.driver.pose_q(pose)
        if orientation is None and not flipped:
            return q
        offsets = self.cfg.handoff.orientation_offsets
        if orientation is not None and orientation not in offsets:
            raise TaskFailed(f"no wrist offset configured for {orientation!r}")
        j6 = self.cfg.joints[-1]
        base = q[-1]
        turn = (offsets[orientation] if orientation is not None else 0.0) + (math.pi if flipped else 0.0)
        options = [base + turn + k * 2 * math.pi for k in (0, -1, 1)]
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
        self.driver.move_to(self._oriented("handoff", r.orientation, self._flipped and self.kin is not None))
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
