"""The scrub-assistant task layer: pick -> verify -> present -> release.

One job runs at a time on a worker thread. The HTTP server (and anything
else) submits jobs and polls `status()`. `release()` and `stop()` are
signals that act on the running job immediately.

Handoff sequence for a request like "needle holder, 6-0 Prolene, backhand":

    hover over slot -> open -> descend -> grasp (check jaws closed on something)
    -> lift -> [tray-cam verify; on mismatch put it back and report]
    -> swing to handoff pose with wrist rotated for forehand/backhand
    -> wait for release (voice "release", API call, or a tug) -> open -> home

Camera-guided pick ("pick caliper"): no slot, the tray camera finds the tool.

    tray_view -> look (two looks must agree) -> plan from the reach map -> hover, wrist turned to the grip angle
    -> descend -> grasp (retry if the jaws closed on nothing) -> lift -> back to tray_view, holding it
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

from .catalog import Catalog, PickRequest, Resolved
from .driver import ArmDriver, ArmFault, MotionAborted
from .pick import GraspPlan, PickError, plan_grasp, wrap180
from .reach import ReachMap
from .verify import Verifier
from .vision import VisionClient, VisionError

log = logging.getLogger(__name__)


class Phase(str, Enum):
    IDLE = "idle"
    LOOKING = "looking"         # at tray_view, asking the camera where the tool is
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
    def __init__(self, driver: ArmDriver, catalog: Catalog, verifier: Verifier,
                 vision: VisionClient | None = None, reach: ReachMap | None = None) -> None:
        self.driver = driver
        self.catalog = catalog
        self.verifier = verifier
        self.vision = vision
        self.reach = reach
        self.cfg = driver.cfg

        self.phase = Phase.IDLE
        self.holding: Resolved | PickRequest | None = None   # PickRequest = picked by camera, no slot
        self._job: dict | None = None
        self._last_job: dict | None = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._release = threading.Event()
        self.events: deque[dict] = deque(maxlen=50)

    # ── public API (thread-safe) ──────────────────────────────────────────

    def submit(self, kind: str, resolved: Resolved | PickRequest | None = None) -> dict:
        fn = {
            "handoff": self._do_handoff,
            "pick": self._do_pick,
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
                "handing": kind == "handoff" or bool(getattr(resolved, "handoff", False)),
            }
            self._job = job
            self._release.clear()
        self.driver.clear_abort()
        threading.Thread(target=self._run, args=(job, fn, resolved), name=f"job-{job['id']}", daemon=True).start()
        return dict(job)

    def release(self) -> dict:
        """Surgeon has the instrument: open the jaws. Works mid-handoff or after a present timeout."""
        with self._lock:
            handing = self._job is not None and self._job["handing"]
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

    def plan_pick(self, req: PickRequest) -> dict:
        """Dry run: look and plan exactly as a pick would, without moving the arm. Point the camera first
        (arm at tray_view, e.g. `go tray_view` in teach)."""
        self._require_reach()
        _, plan = self._look_and_plan(req)
        return plan.describe()

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

    def _do_pick(self, req: PickRequest) -> str:
        pc, d = self.cfg.pick, self.driver
        self._require_reach()
        view = d.poses.poses.get(pc.view_pose)
        if view is None or not (view.taught or d.sim):
            raise TaskFailed(f"the arm has not been taught its '{pc.view_pose}' pose yet")
        if self.holding is not None:
            raise TaskFailed(f"already holding {self.holding.describe()}; release it or return it first")
        slow = self.cfg.motion.approach_speed_scale
        for attempt in range(pc.retries + 1):
            self.phase = Phase.LOOKING
            d.move_pose(pc.view_pose)
            d.gripper_open()
            inst, plan = self._look_and_plan(req)
            self._log(f"picking {inst.name}: px {plan.grasp_px[0]:.0f},{plan.grasp_px[1]:.0f} "
                      f"grip {plan.grip_deg:+.0f} deg ({plan.anchor})")
            self.phase = Phase.PICKING
            d.move_to(plan.above_q)                 # the wrist turns up here, clear of the tray
            d.move_to(plan.grasp_q, speed=slow)
            got_it = d.gripper_grasp()
            d.move_to(plan.above_q, speed=slow)
            if got_it:
                self.holding = PickRequest(candidates=[inst], label=inst.name, handoff=req.handoff)
                break
            d.gripper_open()
            self._log(f"jaws closed on nothing (try {attempt + 1} of {pc.retries + 1})")
        else:
            d.move_pose(pc.view_pose)
            self.phase = Phase.IDLE
            raise TaskFailed(f"I couldn't get a grip on the {req.label} after {pc.retries + 1} tries")
        if req.handoff:
            self._present(self.holding)
            how = self._wait_for_take()
            if how == "timeout":
                return f"presented {inst.name}; not taken, still holding"
            self._release_and_retreat()
            return f"picked {inst.name} and handed it over ({how})"
        self.phase = Phase.MOVING
        d.move_pose(pc.view_pose, speed=slow)
        self.phase = Phase.IDLE
        return f"picked {inst.name}; holding it at {pc.view_pose} (send release to let go)"

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
        if self.holding.slot is None:
            raise TaskFailed("objects picked by camera are not put back yet; use release")
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

    def _require_reach(self) -> None:
        if self.reach is None or self.vision is None:
            raise TaskFailed("camera picking is not set up (pick.vision_url, reach map)")
        if missing := self.reach.ready():
            raise TaskFailed("; ".join(missing) + " (use `reach` in the teach tool)")

    def _look(self, req: PickRequest) -> dict:
        """Ask the camera until two looks in a row agree, so the arm never chases a moving or flickering target."""
        pc = self.cfg.pick
        names = [i.id for i in req.candidates]
        prev = None
        for _ in range(pc.relook_max):
            try:
                det = self.vision.detect(names, pc.frames)
            except VisionError as e:
                raise TaskFailed(str(e)) from e
            if not det.get("found"):
                raise TaskFailed(f"I can't see a {req.label} on the tray")
            if det["stable"] and prev is not None \
                    and math.dist(prev["center_px"], det["center_px"]) <= pc.relook_tol_px \
                    and abs(wrap180(prev["grip_deg"] - det["grip_deg"])) <= 5.0:
                return det
            prev = det if det["stable"] else None
        raise TaskFailed(f"the {req.label} keeps moving, or the camera view is unsteady; not picking")

    def _look_and_plan(self, req: PickRequest):
        det = self._look(req)
        inst = next((i for i in req.candidates if i.id == det.get("name")), req.candidates[0])
        try:
            plan: GraspPlan = plan_grasp(det, self.reach, self.cfg.pick, self.cfg.joints, inst.grasp_along)
            self.driver.check_limits(plan.above_q)
            self.driver.check_limits(plan.grasp_q)
        except (PickError, ArmFault) as e:
            raise TaskFailed(f"can't pick the {req.label}: {e}") from e
        return inst, plan

    def _acquire(self, r: Resolved) -> None:
        """Make sure we're holding the requested slot's instrument, picking it if needed."""
        if self.holding is not None and self.holding.slot is None:
            raise TaskFailed(f"holding {self.holding.describe()}; release it first")
        if self.holding is not None:
            if self.holding.slot.id == r.slot.id:
                self.holding = r          # same instrument; orientation may differ
                self._log(f"already holding {r.describe()}")
                return
            self._log(f"putting back {self.holding.describe()} first")
            self._return_held()
        self._pick(r)
        self._verify(r)

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
        res = self.verifier.verify(r.instrument.id, r.slot.id)
        self._log(f"tray-cam: ok={res.ok} detected={res.detected} conf={res.confidence} {res.note}".strip())
        if not res.ok:
            self._return_held()
            self.phase = Phase.IDLE
            raise TaskFailed(f"wrong instrument in {r.slot.id}: {res.note}")

    def _return_held(self) -> None:
        assert self.holding is not None
        d, slot = self.driver, self.holding.slot
        slow = self.cfg.motion.approach_speed_scale
        self.phase = Phase.RETURNING
        d.move_pose(slot.approach[0])
        d.move_path(slot.approach[1:], speed=slow)
        d.gripper_open()
        self.holding = None
        d.move_path(list(reversed(slot.approach[:-1])), speed=slow)

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
