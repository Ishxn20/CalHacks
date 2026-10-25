"""The console's brain: utterance -> intent -> arm command, and arm status -> ledger, events and audit.

`handle_text` is what voice, the text box and the dashboard all call. `poll` watches the arm's /status and
keeps the ledger honest (an instrument only counts as handed over once the arm says the job finished).
Everything that happens is published as an event and the important parts are written to the audit log.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from typing import Callable

from scrub_arm.audit import AuditLog

from .arm_client import ArmClient, ArmError
from .intent import Intent
from .ledger import HELD, IN_FIELD, TRAY, USED, Ledger, UnknownInstrument
from .report import build_facts, predict_next, summarize_case
from .sinks import MemorySink, Sink
from .wake import WakeGate

log = logging.getLogger(__name__)

METRICS_BATCH_S = 5.0                     # one Convex write per ~5 s of 20 Hz arm readings
HOLDING_PHASES = ("verifying", "presenting", "staged")
ASK_MEMORY_S = 30.0          # how long a clarifying question stays relevant
_STAGE_IT = re.compile(r"^(?:stage|get|hold|grab)\s+(?:it|that|that one)(?:\s+ready)?[.!]?$", re.I)


class Bridge:
    def __init__(self, arm: ArmClient, ledger: Ledger, understand: Callable[[str, str], Intent],
                 sinks: list[Sink] | None = None, audit: AuditLog | None = None, sim: bool = False,
                 observe: Callable[[], list[str] | None] | None = None, triage=None, bitdeer=None) -> None:
        self.arm, self.ledger, self.understand, self.audit = arm, ledger, understand, audit
        self.sim, self.observe, self.triage, self.bitdeer = sim, observe, triage, bitdeer
        self.gate = WakeGate()
        self.stream = MemorySink()
        self.sinks: list[Sink] = [self.stream, *(sinks or [])]
        self._lock = threading.RLock()
        self.status: dict = {}
        self.online: bool | None = None
        self.audit_state: dict = {}
        self._pending: dict[int, dict] = {}       # arm job id -> {"instrument", "kind"}
        self._holding_id: str | None = None       # instrument id the arm is carrying
        self._suggested: str | None = None        # name of the last suggested next instrument ("stage it")
        self._asked: tuple[str, str, float] | None = None   # (what they said, the question we asked, when)
        self._done_job = 0
        self._seen_arm_events: set[tuple] = set()
        self._fault_seen: str | None = None
        self._sim_removed: set[str] = set()
        self._polls = 0
        self.telemetry: deque[dict] = deque(maxlen=150)   # ~1 min of joint readings: Droid's evidence on a fault
        self.observers: list[Callable[[dict], None]] = []  # see every arm status (the dataset recorder)
        self._batch: list[dict] = []              # arm readings not yet sent as a "metrics" event
        self._last_sample_t: float | None = None  # arm clock time of the newest reading seen
        self._emit_ledger()                       # so a fresh Convex board shows the starting tray, not stale data

    # ── plumbing ──────────────────────────────────────────────────────────

    def emit(self, kind: str, /, **data) -> None:
        if kind == "announce":                    # the robot spoke unprompted: let the answer come without "Scrub"
            self.gate.open()
        for s in self.sinks:
            try:
                s.publish(kind, data)
            except Exception:  # noqa: BLE001 - one broken sink must not stop the others
                log.exception("sink %s failed on %s", type(s).__name__, kind)

    def _audit(self, kind: str, /, **data) -> None:
        if self.audit:
            self.audit.append(kind, **data)

    def _emit_ledger(self) -> None:
        self.emit("ledger", items=self.ledger.snapshot(), counts=self.ledger.counts())

    def _move(self, instrument: str | None, to: str) -> None:
        if not instrument:
            return
        try:
            if self.ledger.move(instrument, to):
                self._emit_ledger()
        except UnknownInstrument:
            self.emit("warning", message=f"{instrument} is not in the ledger")

    def context(self) -> str:
        s = self.status
        fix = "yes" if self.triage and self.triage.latest_pending() else "no"
        ctx = f"phase={s.get('phase', 'unknown')}; holding={s.get('holding') or 'nothing'}; fix awaiting decision={fix}"
        if self._asked and time.time() - self._asked[2] < ASK_MEMORY_S:
            ctx += f'; you just asked "{self._asked[1]}" after they said "{self._asked[0]}"'
        return ctx

    def announce(self, text: str) -> None:
        """Something the robot says without being asked (a fault, a failed pick, a fix to decide on)."""
        self.emit("announce", text=text)

    # ── speech in ─────────────────────────────────────────────────────────

    def handle_heard(self, text: str, via: str = "voice") -> dict:
        """Hands-free entry point: everything the always-on mic transcribes comes here. Talk that isn't meant
        for the robot is dropped without a trace (see wake.py)."""
        request = self.gate.check(text, self.status.get("phase"))
        if request is None:
            return {"ignored": True}
        if not request:
            self.gate.open()
            return {"say": "Yes?", "intent": "wake", "ok": True, "heard": text}
        out = self.handle_text(request, via=via)
        self.gate.open()                          # a follow-up ("approve", "backhand") needs no wake word
        return out

    def handle_text(self, text: str, via: str = "text") -> dict:
        if self._suggested and _STAGE_IT.match(text.strip()):
            text = f"stage the {self._suggested}"
        self.emit("utterance", text=text, via=via)
        self._audit("utterance", text=text[:300], via=via)
        intent = self.understand(text, self.context())          # may call Claude: keep it outside the lock
        with self._lock:
            self.emit("intent", kind=intent.kind, fields=intent.fields, say=intent.say, source=intent.source)
            say, ok = intent.say, True
            try:
                say = self._dispatch(intent)
            except ArmError as e:
                say, ok = e.say, False
                self.emit("error", message=str(e), say=e.say)
            self._audit("command", kind=intent.kind, fields=intent.fields, source=intent.source, ok=ok)
            self.emit("say", text=say)
            # Remember a clarifying question for one follow-up, so "Potts" can answer "which scissors?".
            self._asked = (text, say, time.time()) if intent.kind == "ask" else None
            return {"say": say, "intent": intent.kind, "source": intent.source, "ok": ok, "heard": text}

    def _dispatch(self, i: Intent) -> str:
        if i.kind in ("handoff", "stage"):
            resp = (self.arm.handoff if i.kind == "handoff" else self.arm.stage)(i.fields)
            self._pending[resp["job"]["id"]] = {"instrument": resp["instrument"], "kind": i.kind}
            self.emit("command", name=i.kind, instrument=resp["instrument"], slot=resp.get("slot"),
                      orientation=resp.get("orientation"), job=resp["job"]["id"])
            return i.say
        if i.kind in ("release", "stop", "home", "return", "reset"):
            r = self.arm.command(i.kind)
            if "job" in r:
                self._pending[r["job"]["id"]] = {"instrument": self._holding_id, "kind": i.kind}
            self.emit("command", name=i.kind)
            return i.say
        if i.kind == "used":
            self._move(i.fields["instrument"], USED)
            self.emit("command", name="used", instrument=i.fields["instrument"])
            return i.say
        if i.kind == "count":
            return self.count_check()["say"]
        if i.kind in ("approve", "reject"):
            return self._decide(i.kind)
        if i.kind == "report":
            self.case_report()
            return "The case report is on the screen."
        if i.kind == "suggest":
            p = self.suggest_next()["prediction"]
            return f"Probably the {p['name']} next. Say stage it to get it ready." if p else \
                "I don't have a confident suggestion yet."
        return i.say        # ask

    def _decide(self, decision: str) -> str:
        p = self.triage.latest_pending() if self.triage else None
        if p is None:
            return "There's no fix waiting for a decision."
        p = (self.triage.approve if decision == "approve" else self.triage.reject)(p.id, who="voice")
        return {"approved": "Approved. Restart the arm for the change to take effect.",
                "rejected": "Rejected. Nothing was changed.",
                "stale": "The arm's settings changed since that was proposed, so I didn't apply it.",
                "invalid": "That change failed the safety check, so I didn't apply it."}.get(p.status, f"The fix is {p.status}.")

    # ── after the case ────────────────────────────────────────────────────

    def case_report(self) -> dict:
        out = summarize_case(build_facts(self.stream.events, self.ledger.counts(), self.audit_state), self.bitdeer)
        self.emit("report", **out)
        return out

    def suggest_next(self) -> dict:
        history = [e["data"]["instrument"] for e in self.stream.of_kind("command")
                   if e["data"].get("name") == "handoff" and e["data"].get("instrument")]
        remaining = {i.instrument: self.ledger.name(i.instrument) for i in self.ledger.items if i.status == TRAY}
        p = predict_next(history, remaining, self.bitdeer)
        if p:
            self._suggested = p["name"]
            self.emit("prediction", **p)
        return {"prediction": p, "available": self.bitdeer is not None}

    # ── count check ───────────────────────────────────────────────────────

    def sim_visible(self) -> list[str]:
        """What a camera would see in simulation: everything the ledger puts on the table, minus what was removed."""
        return [i.instrument for i in self.ledger.items if i.status in (TRAY, USED) and i.uid not in self._sim_removed]

    def _observed(self) -> tuple[list[str] | None, str]:
        if self.sim:
            return (self.observe() if self.observe else self.sim_visible()), "simulated"
        if self.observe:
            try:
                return self.observe(), "camera"
            except Exception as e:  # noqa: BLE001
                log.warning("visual count failed: %s", e)
        return None, "ledger"

    def count_check(self) -> dict:
        with self._lock:
            observed, source = self._observed()
            rec = self.ledger.reconcile(observed)
            say = self.ledger.say_for(rec)
            self.emit("count", **rec.to_dict(), say=say, source=source)
            self._audit("count", ok=rec.ok, missing=rec.missing, outstanding=rec.outstanding, source=source)
            return {**rec.to_dict(), "say": say, "source": source}

    def sim_remove(self, instrument: str) -> bool:
        """Simulation only: someone takes an instrument off the table without telling the console."""
        if not self.sim:
            raise PermissionError("only available in simulation")
        with self._lock:
            for item in self.ledger.items:
                if item.instrument == instrument and item.status in (TRAY, USED) and item.uid not in self._sim_removed:
                    self._sim_removed.add(item.uid)
                    self.emit("sim", action="removed", instrument=instrument)
                    return True
        return False

    def sim_restore(self) -> None:
        if not self.sim:
            raise PermissionError("only available in simulation")
        with self._lock:
            self._sim_removed.clear()
            self.emit("sim", action="restored")

    # ── arm status in ─────────────────────────────────────────────────────

    def poll(self) -> None:
        try:
            st = self.arm.status(since=self._last_sample_t or 0.0)
        except ArmError as e:
            if self.online is not False:
                self.online = False
                self.emit("arm_offline", message=str(e))
            return
        with self._lock:
            if self.online is not True:
                self.online = True
                self.emit("arm_online")
            self._polls += 1
            self._track(st)
            if self._polls % 10 == 1:
                self.anchor()

    def _track(self, st: dict) -> None:
        old, self.status = self.status, st
        arm = st.get("arm") or {}
        self.telemetry.append({"t": round(time.time(), 2), "phase": st.get("phase"), "q": arm.get("q_deg") or [],
                               "q_cmd": arm.get("q_cmd_deg") or [], "gripper": arm.get("gripper_deg")})
        self._collect(st)
        for observe in self.observers:
            try:
                observe(st)
            except Exception:  # noqa: BLE001 - an observer must never stop the arm watch
                log.exception("status observer failed")
        key = lambda s: (s.get("phase"), s.get("holding"), (s.get("arm") or {}).get("fault"))  # noqa: E731
        if key(old) != key(st):
            self.emit("arm", phase=st.get("phase"), holding=st.get("holding"), fault=arm.get("fault"),
                      enabled=arm.get("enabled"))
        for e in st.get("events") or []:
            k = (e.get("t"), e.get("msg"))
            if k not in self._seen_arm_events:
                self._seen_arm_events.add(k)
                if old:                                   # don't replay history on the first poll
                    self.emit("arm_event", message=e.get("msg"))
        job = st.get("job")
        if st.get("phase") in HOLDING_PHASES and job and job.get("id") in self._pending:
            p = self._pending[job["id"]]
            if p["instrument"] and self._holding_id != p["instrument"]:
                self._holding_id = p["instrument"]
                self._move(p["instrument"], HELD)
        last = st.get("last_job")
        if last and last["id"] > self._done_job and last.get("status") != "running":
            self._done_job = last["id"]
            self._finish(last, st)
        fault = arm.get("fault") or (last.get("error") if st.get("phase") == "fault" and last else None)
        if fault and fault != self._fault_seen:
            self._fault_seen = fault
            self.emit("fault", message=fault)
            self._audit("fault", message=fault)
            self.announce(f"I've stopped. {fault}")
            if self.triage:
                self.triage.trigger(fault, self.recent_context())
        elif not fault:
            self._fault_seen = None

    def _collect(self, st: dict) -> None:
        """Batch the arm's 20 Hz readings into one "metrics" event per ~5 s (stored in Convex, not streamed)."""
        samples = st.get("samples")
        if not samples:
            return
        if self._last_sample_t is not None:      # the first poll only learns where the arm's clock is
            stamp = {"phase": st.get("phase") or "", "job": (st.get("job") or {}).get("id"),
                     "fault": (st.get("arm") or {}).get("fault")}
            self._batch += [{**s, **stamp} for s in samples]
        self._last_sample_t = samples[-1]["t"]
        if self._batch and self._batch[-1]["t"] - self._batch[0]["t"] >= METRICS_BATCH_S:
            self.emit("metrics", t0=self._batch[0]["t"], t1=self._batch[-1]["t"], source="sim" if self.sim else "arm",
                      samples=self._batch)
            self._batch = []

    def _finish(self, job: dict, st: dict) -> None:
        p = self._pending.pop(job["id"], {})
        inst, kind, ok = p.get("instrument") or self._holding_id, job["kind"], job["status"] == "done"
        result = job.get("result") or ""
        self.emit("job", id=job["id"], kind=kind, status=job["status"], result=result, error=job.get("error"))
        self._audit("job", id=job["id"], kind=kind, status=job["status"], result=result[:200], error=job.get("error"))
        if kind in ("handoff", "stage") and ok:
            if inst and self._holding_id != inst:
                self._holding_id = inst
                self._move(inst, HELD)
            if result.startswith("handed"):
                self._hand_over()
        elif kind == "release" and ok:
            self._hand_over()
        elif kind in ("handoff", "stage", "return") and not st.get("holding"):
            self._move(self._holding_id or inst, TRAY)         # failed pick or put-back: it is on the tray again
            self._holding_id = None
        if kind in ("handoff", "stage") and job["status"] == "failed":
            self.announce(f"That didn't work: {job.get('error') or 'the pick failed'}.")

    def _hand_over(self) -> None:
        self._move(self._holding_id, IN_FIELD)
        self._holding_id = None

    # ── audit anchoring ───────────────────────────────────────────────────

    def anchor(self) -> dict:
        """Publish the audit log heads outside the arm's own disk, so cutting entries off the end shows up."""
        state = {}
        if self.audit:
            v = self.audit.verify()
            state["console"] = {"ok": v.ok, "entries": v.entries, "head": v.head, "error": v.error}
        try:
            a = self.arm.audit()
            state["arm"] = {"ok": a.get("ok"), "entries": a.get("entries"), "head": a.get("head"), "error": a.get("error")}
        except ArmError as e:
            state["arm"] = {"ok": None, "error": str(e)}
        self.audit_state = state
        for source, s in state.items():
            self.emit("audit", source=source, seq=s.get("entries") or 0, head=s.get("head") or "", ok=bool(s.get("ok")))
        return state

    # ── views ─────────────────────────────────────────────────────────────

    def recent_context(self) -> dict:
        return {"status": self.status, "telemetry": list(self.telemetry),
                "events": [e for e in self.stream.events[-60:] if e["kind"] in ("arm_event", "job", "fault", "command", "arm")]}

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "sim": self.sim, "online": self.online, "arm": self.status,
                "ledger": {"items": self.ledger.snapshot(), "counts": self.ledger.counts()},
                "events": self.stream.events[-120:], "audit": self.audit_state,
                "triage": self.triage.snapshot() if self.triage else [], "seq": self.stream._seq,
            }

    def run(self, interval_s: float = 0.4, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        while not stop.is_set():
            try:
                self.poll()
            except Exception:  # noqa: BLE001 - keep watching the arm whatever happens
                log.exception("poll failed")
            stop.wait(interval_s)
