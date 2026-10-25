import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import droid_sdk
from droid_sdk import (AssistantMessage, EditAction, ExecuteAction, PermissionOption, PermissionRequest,
                       ToolCall, ToolConfirmationOutcome, ToolResult, ToolUseBlock)
from scrub_arm.audit import AuditLog
from scrub_console.triage import (ALLOWED, RehearsalRunner, SdkDroidRunner, Triage, make_permission_handler,
                                  validate_change)

ARM_YAML = (Path(__file__).resolve().parents[2] / "arm" / "config" / "arm.yaml").read_text()
FAULT = "joint3 is 28.0 deg off its setpoint (collision or blocked?) - arm frozen in place"


def edited(**changes):
    """arm.yaml with `key: value` lines replaced (keys are the last part of a dotted name)."""
    text = ARM_YAML
    for key, val in changes.items():
        lines = text.splitlines()
        i = next(n for n, l in enumerate(lines) if l.strip().startswith(key + ":"))
        head = lines[i].split(":")[0]
        lines[i] = f"{head}: {val}"
        text = "\n".join(lines) + "\n"
    return text


# ── validation ─────────────────────────────────────────────────────────────

def test_a_tuning_change_inside_bounds_is_accepted():
    v = validate_change(ARM_YAML, edited(tracking_error_limit_deg=30.0, contact_deviation_deg=2.0))
    assert v.ok and {c["key"] for c in v.changes} == {"motion.tracking_error_limit_deg", "pick.contact_deviation_deg"}


@pytest.mark.parametrize("key,val", [
    ("tracking_error_limit_deg", 90.0),            # above the allowed range
    ("tracking_error_limit_deg", "null"),          # would switch the collision guard off
    ("tracking_error_limit_deg", "'25'"),          # not a number
    ("grip_torque_nm", 3.0),                       # would crush the instrument
    ("calibrated", "false"),                       # not on the list
    ("open_deg", 12.0),                            # gripper calibration
    ("channel", "can9"),
])
def test_anything_else_is_rejected(key, val):
    v = validate_change(ARM_YAML, edited(**{key: val}))
    assert not v.ok and v.errors


def test_joint_limits_can_never_change_through_triage():
    text = ARM_YAML.replace("limits_deg: [-160, 160]", "limits_deg: [-200, 200]", 1)
    v = validate_change(ARM_YAML, text)
    assert not v.ok and any("joint" in e or "limits" in e for e in v.errors)


def test_no_change_added_keys_and_broken_yaml_are_not_proposals():
    assert not validate_change(ARM_YAML, ARM_YAML).ok
    assert not validate_change(ARM_YAML, ARM_YAML + "\nextra_setting: 1\n").ok
    assert not validate_change(ARM_YAML, "motion: [unclosed").ok


def test_allowed_keys_exist_in_the_real_config():
    from scrub_console.triage import flatten

    assert set(ALLOWED) <= set(flatten(yaml.safe_load(ARM_YAML)))


# ── the triage flow ────────────────────────────────────────────────────────

class FakeRunner:
    def __init__(self, new_text=None, boom=None):
        self.new_text, self.boom, self.prompts, self.sandboxes = new_text, boom, [], []

    async def run(self, sandbox, prompt, on_step):
        self.prompts.append(prompt)
        self.sandboxes.append(sandbox)
        on_step({"type": "tool", "name": "Read", "input": {"file_path": str(sandbox / "arm.yaml")}})
        if self.boom:
            raise self.boom
        if self.new_text:
            (sandbox / "arm.yaml").write_text(self.new_text)
        return "Raised the guard threshold: the joint lagged briefly under load."


@pytest.fixture
def rig(tmp_path):
    cfg = tmp_path / "arm.yaml"
    cfg.write_text(ARM_YAML)
    emitted, audit = [], AuditLog(tmp_path / "audit.jsonl")

    def make(runner):
        return Triage(cfg, runner, tmp_path / "triage", emit=lambda kind, /, **d: emitted.append((kind, d)),
                      audit=audit, background=False)

    return SimpleNamespace(cfg=cfg, make=make, emitted=emitted, audit=audit, tmp=tmp_path)


def test_droid_gets_a_sandbox_copy_and_the_fault_not_the_repo(rig):
    runner = FakeRunner(edited(tracking_error_limit_deg=30.0))
    t = rig.make(runner)
    telemetry = [{"t": 100.0, "phase": "moving", "q": [0, 20, 40, 0, 0, 0], "q_cmd": [0, 20, 40, 0, 0, 0], "gripper": 259.4},
                 {"t": 100.4, "phase": "fault", "q": [0, 21, 12, 0, 0, 0], "q_cmd": [0, 22, 40, 0, 0, 0], "gripper": 259.4}]
    pid = t.trigger(FAULT, {"events": [{"msg": "x"}], "telemetry": telemetry})
    sandbox = runner.sandboxes[0]
    assert sandbox != rig.cfg.parent and (sandbox / "fault_context.json").exists()
    assert json.loads((sandbox / "fault_context.json").read_text())["fault"] == FAULT
    csv = (sandbox / "telemetry.csv").read_text().splitlines()
    assert csv[0].startswith("time_s,phase,q1") and csv[2].startswith("0.4,fault,0,21,12")    # joint 3 lagging 28 deg
    assert "tracking_error_limit" in (sandbox / "driver.py").read_text()                     # the guard's source
    assert FAULT in runner.prompts[0] and "SMALLEST edit" in runner.prompts[0]
    assert rig.cfg.read_text() == ARM_YAML                       # the real config is untouched
    p = t.proposals[pid]
    assert p.status == "pending" and "+" in p.diff and p.changes[0]["new"] == 30.0
    assert [s["name"] for s in p.steps] == ["Read"]
    assert {"proposal", "droid_step"} <= {k for k, _ in rig.emitted}


def test_approving_writes_the_file_backs_it_up_and_is_audited(rig):
    t = rig.make(FakeRunner(edited(tracking_error_limit_deg=30.0)))
    pid = t.trigger(FAULT, {})
    p = t.approve(pid, who="dr_lee")
    assert p.status == "approved" and p.needs_restart
    assert yaml.safe_load(rig.cfg.read_text())["motion"]["tracking_error_limit_deg"] == 30.0
    backups = list((rig.tmp / "triage").glob("arm.yaml.*.bak"))
    assert len(backups) == 1 and backups[0].read_text() == ARM_YAML
    logged = [json.loads(l) for l in rig.audit.path.read_text().splitlines()]
    assert [e["kind"] for e in logged] == ["triage_proposal", "triage_decision"]
    assert logged[1]["data"]["by"] == "dr_lee" and logged[1]["data"]["decision"] == "approved"


def test_rejecting_changes_nothing(rig):
    t = rig.make(FakeRunner(edited(tracking_error_limit_deg=30.0)))
    pid = t.trigger(FAULT, {})
    assert t.reject(pid).status == "rejected" and rig.cfg.read_text() == ARM_YAML
    with pytest.raises(ValueError):
        t.approve(pid)                                            # a decision is final


def test_a_proposal_outside_the_allowlist_is_invalid_and_cannot_be_approved(rig):
    t = rig.make(FakeRunner(edited(tracking_error_limit_deg="null", calibrated="false")))
    pid = t.trigger(FAULT, {})
    p = t.proposals[pid]
    assert p.status == "invalid" and p.errors
    with pytest.raises(ValueError):
        t.approve(pid)
    assert rig.cfg.read_text() == ARM_YAML


def test_a_config_that_changed_since_the_proposal_is_stale(rig):
    t = rig.make(FakeRunner(edited(tracking_error_limit_deg=30.0)))
    pid = t.trigger(FAULT, {})
    rig.cfg.write_text(ARM_YAML.replace("speed_scale: 0.5", "speed_scale: 0.4", 1))
    assert t.approve(pid).status == "stale"
    assert "speed_scale: 0.4" in rig.cfg.read_text()              # the operator's newer edit survives


def test_droid_changing_nothing_is_a_valid_outcome(rig):
    t = rig.make(FakeRunner(None))
    p = t.proposals[t.trigger(FAULT, {})]
    assert p.status == "no_change" and p.summary and not p.diff


def test_without_droid_installed_it_says_so(rig):
    t = rig.make(None)
    p = t.proposals[t.trigger(FAULT, {})]
    assert p.status == "unavailable" and "Droid" in p.summary


def test_a_crashing_runner_never_escapes(rig):
    t = rig.make(FakeRunner(boom=RuntimeError("droid process died")))
    p = t.proposals[t.trigger(FAULT, {})]
    assert p.status == "error" and "died" in p.summary


def test_one_triage_run_at_a_time(rig):
    t = rig.make(FakeRunner())
    t._busy = True
    assert t.trigger(FAULT, {}) is None and ("triage_skipped" in {k for k, _ in rig.emitted})


def test_the_rehearsal_runner_is_labelled_simulated(rig):
    t = rig.make(RehearsalRunner())
    p = t.proposals[t.trigger(FAULT, {})]
    assert p.simulated and p.status == "pending" and "REHEARSAL" in p.summary
    assert t.snapshot()[0]["simulated"] is True


# ── the permission handler, with the SDK's own types ───────────────────────

def _request(action, offered=(ToolConfirmationOutcome.PROCEED_ONCE, ToolConfirmationOutcome.CANCEL)):
    return PermissionRequest(actions=[action], options=[PermissionOption(label=o.value, value=o) for o in offered],
                             associated_session_ids=["s"])


def _edit(path):
    return EditAction(tool_use=ToolUseBlock(id="1", name="Edit", input={}, thought_signature=None), file_path=str(path),
                      file_name=Path(path).name, old_content="a", new_content="b")


def test_permission_handler_allows_only_the_sandbox_config(tmp_path):
    h = make_permission_handler(tmp_path)
    assert h(_request(_edit(tmp_path / "arm.yaml"))).selected_option == ToolConfirmationOutcome.PROCEED_ONCE
    assert h(_request(_edit(tmp_path / "other.yaml"))).selected_option == ToolConfirmationOutcome.CANCEL
    assert h(_request(_edit(tmp_path / ".." / "arm.yaml"))).selected_option == ToolConfirmationOutcome.CANCEL
    run = ExecuteAction(tool_use=ToolUseBlock(id="2", name="Execute", input={}, thought_signature=None),
                        full_command="rm -rf /", command="rm", extracted_commands=None, impact_level=None,
                        risk_level_reason=None)
    assert h(_request(run)).selected_option == ToolConfirmationOutcome.CANCEL
    only_cancel = _request(_edit(tmp_path / "arm.yaml"), offered=(ToolConfirmationOutcome.CANCEL,))
    assert h(only_cancel).selected_option == ToolConfirmationOutcome.CANCEL


# ── SdkDroidRunner against a faked Session ─────────────────────────────────

class FakeStream:
    def __init__(self, events, result):
        self._events, self.result = events, result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e
        return gen()


def test_sdk_runner_restricts_tools_and_reports_every_step(tmp_path, monkeypatch):
    seen = {}
    now = datetime.now(timezone.utc)
    events = [ToolCall(name="Read", tool_use_id="1", input={"file_path": "arm.yaml"}),
              ToolResult(tool_use_id="1", tool_name="Read", content="motion: ...", is_error=False),
              AssistantMessage(id="m", content=(), text="Looks like a transient lag.", parent_id=None, created_at=now, updated_at=now)]

    class FakeSession:
        def __init__(self, **kw):
            seen.update(kw)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        def stream(self, prompt, **kw):
            seen["prompt"], seen["stream_kw"] = prompt, kw
            return FakeStream(events, SimpleNamespace(success=True, text="Raised the threshold.", subtype="success"))

    monkeypatch.setattr(droid_sdk, "Session", FakeSession)
    steps = []
    out = asyncio.run(SdkDroidRunner(timeout_s=30).run(tmp_path, "fix it", steps.append))
    assert out == "Raised the threshold."
    assert [s["type"] for s in steps] == ["tool", "result", "message"]
    cfg = seen["config"]
    assert set(cfg.restrict_tools) == {"Read", "Edit", "Grep", "Glob"} and cfg.autonomy == droid_sdk.Autonomy.LOW
    assert seen["cwd"] == str(tmp_path) and seen["interactions"].on_permission is not None
    assert seen["stream_kw"]["timeout"] == 30


def test_sdk_runner_reports_a_run_that_did_not_finish(tmp_path, monkeypatch):
    class FakeSession:
        def __init__(self, **kw): ...
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def stream(self, prompt, **kw):
            return FakeStream([], SimpleNamespace(success=False, text="", subtype="interrupted"))

    monkeypatch.setattr(droid_sdk, "Session", FakeSession)
    assert "interrupted" in asyncio.run(SdkDroidRunner().run(tmp_path, "x", lambda s: None))
