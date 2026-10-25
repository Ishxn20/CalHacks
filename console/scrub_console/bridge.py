"""The console's brain: utterance -> intent -> arm command, and arm status -> ledger, events and audit.

`handle_text` is what voice, the text box and the dashboard all call. `poll` watches the arm's /status and
keeps the ledger honest (an instrument only counts as handed over once the arm says the job finished).
Everything that happens is published as an event and the important parts are written to the audit log.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from scrub_arm.audit import AuditLog

from .arm_client import ArmClient, ArmError
from .intent import Intent
from .ledger import HELD, IN_FIELD, TRAY, USED, Ledger, UnknownInstrument
from .sinks import MemorySink, Sink

log = logging.getLogger(__name__)

HOLDING_PHASES = ("verifying", "presenting", "staged")


class Bridge:
    def __init__(self, arm: ArmClient, ledger: Ledger, understand: Callable[[str, str], Intent],
                 sinks: list[Sink] | None = None, audit: AuditLog | None = None, sim: bool = False,
                 observe: Callable[[], list[str] | None] | None = None, triage=None) -> None:
        self.arm, self.ledger, self.understand, self.audit = arm, ledger, understand, audit
        self.sim, self.observe, self.triage = sim, observe, triage
        self.stream = MemorySink()
        self.sinks: list[Sink] = [self.stream, *(sinks or [])]
        self._lock = threading.RLock()
        self.status: dict = {}
        self.online: bool | None = None
        self.audit_state: dict = {}
        self._pending: dict[int, dict] = {}       # arm job id -> {"instrument", "kind"}
        self._holding_id: str | None = None       # instrument id the arm is carrying
        self._done_job = 0
        self._seen_arm_events: set[tuple] = set()
        self._fault_seen: str | None = None
        self._sim_removed: set[str] = set()
        self._polls = 0

    # ── plumbing ──────────────────────────────────────────────────────────

    def emit(self, kind: str, /, **data) -> None:
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
        return f"phase={s.get('phase', 'unknown')}; holding={s.get('holding') or 'nothing'}"

    # ── speech in ─────────────────────────────────────────────────────────

    def handle_text(self, text: str, via: str = "text") -> dict:
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
        return i.say        # ask

    # ── count check ───────────────────────────────────────────────────────

    def _observed(self) -> tuple[list[str] | None, str]:
        if self.sim:
            return ([i.instrument for i in self.ledger.items
                     if i.status in (TRAY, USED) and i.uid not in self._sim_removed], "simulated")
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
            st = self.arm.status()
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
            if self.triage:
                self.triage.trigger(fault, self.recent_context())
        elif not fault:
            self._fault_seen = None

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
        return {"status": self.status, "events": [e for e in self.stream.events[-60:]
                                                  if e["kind"] in ("arm_event", "job", "fault", "command", "arm")]}

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
