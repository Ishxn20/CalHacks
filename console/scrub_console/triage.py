"""Fault triage with Factory Droid: an agent that reads what went wrong and proposes a config fix, for a human to approve.

Trigger: the arm reports a fault (collision guard, CAN errors...). The bridge calls `Triage.trigger`.
What Droid can touch, by construction:
  - It works in a throwaway sandbox folder holding a COPY of arm.yaml and the fault context, never the repo.
  - Its tools are restricted to Read / Edit / Grep / Glob, and a permission handler allows an edit of exactly
    that one sandbox file and refuses everything else (commands, new files, patches elsewhere).
  - Whatever it changes is diffed against the real arm.yaml and checked against ALLOWED: only listed tuning
    values, only numbers, only inside the listed bounds. Joint limits, gripper calibration, motor IDs and
    poses can never be changed this way, and the collision guard can be loosened but never switched off.
  - Nothing is applied until a person approves the diff; approving re-checks that arm.yaml has not changed
    since the proposal, backs it up, and writes it atomically. The arm reads its config at startup, so the
    server has to be restarted for a change to take effect.
Every step Droid takes (tool calls, results, messages) is published as an event so the dashboard can show it.
"""
from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import yaml

log = logging.getLogger(__name__)

# dotted key in arm.yaml -> (lowest, highest) value a proposal may set
ALLOWED: dict[str, tuple[float, float]] = {
    "motion.tracking_error_limit_deg": (5.0, 40.0),
    "motion.speed_scale": (0.05, 1.0),
    "motion.approach_speed_scale": (0.05, 1.0),
    "motion.settle_timeout_s": (0.5, 5.0),
    "pick.contact_deviation_deg": (0.5, 5.0),
    "pick.descend_step_mm": (1.0, 6.0),
    "pick.hover_mm": (40.0, 150.0),
    "gripper.grip_torque_nm": (0.3, 1.2),
}
SANDBOX_FILE = "arm.yaml"


def flatten(d, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


@dataclass
class Validation:
    ok: bool
    changes: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def validate_change(original: str, proposed: str) -> Validation:
    """Is `proposed` an acceptable edit of `original`? Only ALLOWED numeric values may differ, and only within bounds."""
    try:
        a, b = flatten(yaml.safe_load(original)), flatten(yaml.safe_load(proposed))
    except yaml.YAMLError as e:
        return Validation(False, errors=[f"proposed file is not valid YAML: {e}"])
    changes, errors = [], []
    for key in sorted(set(a) | set(b)):
        if key in a and key in b and a[key] == b[key]:
            continue
        if key not in a or key not in b:
            errors.append(f"{key}: added or removed")
            continue
        new = b[key]
        if key not in ALLOWED:
            errors.append(f"{key}: not a value Droid may change")
        elif isinstance(new, bool) or not isinstance(new, (int, float)):
            errors.append(f"{key}: must be a number, got {new!r}")
        elif not ALLOWED[key][0] <= new <= ALLOWED[key][1]:
            errors.append(f"{key}: {new} is outside the allowed range {ALLOWED[key][0]}..{ALLOWED[key][1]}")
        else:
            changes.append({"key": key, "old": a[key], "new": new})
    return Validation(not errors and bool(changes), changes, errors)


def make_diff(original: str, proposed: str) -> str:
    return "".join(difflib.unified_diff(original.splitlines(True), proposed.splitlines(True),
                                        "arm.yaml (current)", "arm.yaml (proposed)"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass
class Proposal:
    id: str
    fault: str
    created: float
    simulated: bool = False
    status: str = "running"        # running | pending | no_change | invalid | unavailable | approved | rejected | stale | error
    summary: str = ""
    diff: str = ""
    changes: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    original_sha: str = ""
    proposed_text: str = ""
    needs_restart: bool = False

    def public(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "proposed_text"}
        d["steps"] = d["steps"][-40:]
        return d


class DroidRunner(Protocol):
    async def run(self, sandbox: Path, prompt: str, on_step: Callable[[dict], None]) -> str: ...


def build_prompt(fault: str) -> str:
    return (
        f'A fault stopped the robot arm: "{fault}"\n\n'
        "This folder holds fault_context.json (the arm's recent events and status) and arm.yaml (its tuning "
        "config). Read both. If a tuning value in arm.yaml plausibly caused this fault or would have "
        "prevented it, make the SMALLEST edit to arm.yaml that addresses it, changing only values under "
        "`motion:`, `pick:`, or `gripper.grip_torque_nm`. Never touch joint limits, gripper calibration, "
        "motor ids, or anything else. If no config change is warranted (for example a real collision), "
        "change nothing. Finish with two or three plain sentences: what you found and why this change."
    )


# ── Droid runner ───────────────────────────────────────────────────────────

def make_permission_handler(sandbox: Path):
    """Allow an edit of the one sandbox file; refuse everything else (commands, new files, other paths)."""
    from droid_sdk import EditAction, PermissionResponse, ToolConfirmationOutcome

    target = (sandbox / SANDBOX_FILE).resolve()

    def allowed(action) -> bool:
        return isinstance(action, EditAction) and Path(action.file_path).resolve() == target

    def handler(request) -> PermissionResponse:
        offered = {o.value for o in request.options}
        ok = bool(request.actions) and all(allowed(a) for a in request.actions)
        choice = ToolConfirmationOutcome.PROCEED_ONCE if ok and ToolConfirmationOutcome.PROCEED_ONCE in offered \
            else ToolConfirmationOutcome.CANCEL
        return PermissionResponse(selected_option=choice, comment=None if ok else "outside the allowed edit", edited_spec_content=None)

    return handler


class SdkDroidRunner:
    """Runs the prompt in a local Droid session (needs the `droid` CLI on PATH and FACTORY_API_KEY or a Droid login)."""

    def __init__(self, timeout_s: float = 180.0, api_key: str | None = None) -> None:
        self.timeout_s, self.api_key = timeout_s, api_key

    async def run(self, sandbox: Path, prompt: str, on_step: Callable[[dict], None]) -> str:
        from droid_sdk import (AssistantMessage, Autonomy, InteractionHandlers, Session, SessionConfig, ToolCall,
                               ToolResult)

        config = SessionConfig(
            autonomy=Autonomy.LOW, restrict_tools=["Read", "Edit", "Grep", "Glob"],
            system_prompt={"type": "preset", "preset": "droid",
                           "append": "You are a fault-triage agent for a robot arm. You may only read the files "
                                     "in this folder and edit arm.yaml. Be brief."},
        )
        handlers = InteractionHandlers(on_permission=make_permission_handler(sandbox))
        async with Session(cwd=str(sandbox), config=config, interactions=handlers, api_key=self.api_key) as session:
            async with session.stream(prompt, timeout=self.timeout_s) as stream:
                async for event in stream:
                    if isinstance(event, ToolCall):
                        on_step({"type": "tool", "name": event.name, "input": dict(event.input)})
                    elif isinstance(event, ToolResult):
                        on_step({"type": "result", "name": event.tool_name, "error": event.is_error,
                                 "preview": str(event.content)[:300]})
                    elif isinstance(event, AssistantMessage) and event.text.strip():
                        on_step({"type": "message", "text": event.text.strip()})
            result = stream.result
        return result.text if result.success else f"Droid did not finish ({result.subtype})."


class RehearsalRunner:
    """NOT an agent: applies a canned tweak so the dashboard flow can be rehearsed without a Droid install.
    Proposals made with it are marked simulated everywhere."""

    simulated = True

    async def run(self, sandbox: Path, prompt: str, on_step: Callable[[dict], None]) -> str:
        path = sandbox / SANDBOX_FILE
        on_step({"type": "tool", "name": "Read", "input": {"file_path": str(path)}})
        cfg = yaml.safe_load(path.read_text())
        old = cfg["motion"]["tracking_error_limit_deg"]
        text = path.read_text().replace(f"tracking_error_limit_deg: {old}", f"tracking_error_limit_deg: {min(old + 5, 40)}", 1)
        on_step({"type": "tool", "name": "Edit", "input": {"file_path": str(path)}})
        path.write_text(text)
        return "REHEARSAL ONLY (no agent ran): raised the collision-guard threshold by 5 degrees."


# ── Triage ─────────────────────────────────────────────────────────────────

class Triage:
    def __init__(self, config_path: Path, runner: DroidRunner | None, work_dir: Path,
                 emit: Callable[..., None] | None = None, audit=None, background: bool = True) -> None:
        self.config_path, self.runner, self.work_dir = Path(config_path), runner, Path(work_dir)
        self.emit = emit or (lambda kind, /, **data: None)
        self.audit, self.background = audit, background
        self.proposals: dict[str, Proposal] = {}
        self._lock = threading.Lock()
        self._busy = False

    # ── trigger ───────────────────────────────────────────────────────────

    def trigger(self, fault: str, context: dict, simulated: bool = False) -> str | None:
        """Start a triage run for this fault. Returns the proposal id, or None if one is already running."""
        with self._lock:
            if self._busy:
                self.emit("triage_skipped", fault=fault, reason="a triage run is already in progress")
                return None
            self._busy = True
        p = Proposal(id=f"t{int(time.time() * 1000) % 10**9}", fault=fault, created=time.time(),
                     simulated=simulated or bool(getattr(self.runner, "simulated", False)))
        self.proposals[p.id] = p
        self._publish(p)
        if self.background:
            threading.Thread(target=self._guarded, args=(p, context), name="triage", daemon=True).start()
        else:
            self._guarded(p, context)
        return p.id

    def _guarded(self, p: Proposal, context: dict) -> None:
        try:
            self._work(p, context)
        except Exception as e:  # noqa: BLE001 - triage trouble must never reach the arm
            log.exception("triage failed")
            p.status, p.summary = "error", f"{type(e).__name__}: {e}"
        finally:
            self._busy = False
            self._publish(p)

    def _step(self, p: Proposal, step: dict) -> None:
        p.steps.append({**step, "t": round(time.time(), 2)})
        self.emit("droid_step", proposal=p.id, **step)

    def _work(self, p: Proposal, context: dict) -> None:
        original = self.config_path.read_text()
        p.original_sha = sha(original)
        if self.runner is None:
            p.status = "unavailable"
            p.summary = ("Droid isn't set up here: install the Droid CLI and set FACTORY_API_KEY "
                         "(or run with --rehearse-droid to rehearse the flow).")
            return
        sandbox = self.work_dir / p.id
        sandbox.mkdir(parents=True, exist_ok=True)
        (sandbox / SANDBOX_FILE).write_text(original)
        (sandbox / "fault_context.json").write_text(json.dumps({"fault": p.fault, **context}, indent=1, default=str))
        summary = asyncio.run(self.runner.run(sandbox, build_prompt(p.fault), lambda s: self._step(p, s)))
        p.summary = summary
        proposed = (sandbox / SANDBOX_FILE).read_text()
        if proposed == original:
            p.status = "no_change"
            return
        v = validate_change(original, proposed)
        p.diff, p.changes, p.errors = make_diff(original, proposed), v.changes, v.errors
        p.proposed_text = proposed
        p.status = "pending" if v.ok else "invalid"
        if self.audit:
            self.audit.append("triage_proposal", id=p.id, status=p.status, changes=v.changes, errors=v.errors,
                              simulated=p.simulated)

    def _publish(self, p: Proposal) -> None:
        self.emit("proposal", **p.public())

    # ── decisions ─────────────────────────────────────────────────────────

    def approve(self, proposal_id: str, who: str = "operator") -> Proposal:
        p = self._pending(proposal_id)
        current = self.config_path.read_text()
        if sha(current) != p.original_sha:
            p.status, p.summary = "stale", "arm.yaml changed after this was proposed; run triage again."
        else:
            v = validate_change(current, p.proposed_text)       # never trust the stored result: check again
            if not v.ok:
                p.status, p.errors = "invalid", v.errors
            else:
                backup = self.work_dir / f"arm.yaml.{int(time.time())}.bak"
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.config_path, backup)
                fd, tmp = tempfile.mkstemp(dir=self.config_path.parent, prefix=".arm.yaml.", suffix=".tmp")
                with os.fdopen(fd, "w") as f:
                    f.write(p.proposed_text)
                os.replace(tmp, self.config_path)
                p.status, p.needs_restart = "approved", True
        if self.audit:
            self.audit.append("triage_decision", id=p.id, decision=p.status, by=who, changes=p.changes)
        self._publish(p)
        return p

    def reject(self, proposal_id: str, who: str = "operator") -> Proposal:
        p = self._pending(proposal_id)
        p.status = "rejected"
        if self.audit:
            self.audit.append("triage_decision", id=p.id, decision="rejected", by=who)
        self._publish(p)
        return p

    def _pending(self, proposal_id: str) -> Proposal:
        p = self.proposals.get(proposal_id)
        if p is None:
            raise KeyError(f"no proposal {proposal_id}")
        if p.status != "pending":
            raise ValueError(f"proposal {proposal_id} is {p.status}, not pending")
        return p

    def snapshot(self) -> list[dict]:
        return [p.public() for p in sorted(self.proposals.values(), key=lambda p: -p.created)][:10]
