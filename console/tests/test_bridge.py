import pytest
from conftest import wait_for

from scrub_arm.audit import AuditLog
from scrub_arm.config import load_tray_config
from scrub_console.arm_client import ArmClient, ArmError
from scrub_console.bridge import Bridge
from scrub_console.intent import understand
from scrub_console.ledger import HELD, IN_FIELD, TRAY, USED, Ledger


@pytest.fixture
def rig(arm_server, tmp_path):
    arm = ArmClient(arm_server.url, arm_server.token)
    ledger = Ledger.from_tray(load_tray_config())
    audit = AuditLog(tmp_path / "console_audit.jsonl")
    tray = load_tray_config()
    bridge = Bridge(arm, ledger, lambda text, ctx: understand(text, arm.resolve, None, ctx, tray), audit=audit, sim=True)
    bridge.poll()                       # first poll: don't treat existing arm history as news
    return bridge, arm_server


def status_of(bridge, instrument):
    return next(i["status"] for i in bridge.ledger.snapshot() if i["instrument"] == instrument)


def kinds(bridge):
    return [e["kind"] for e in bridge.stream.events]


def test_a_full_case_from_request_to_a_correct_count(rig):
    bridge, _ = rig
    r = bridge.handle_text("needle holder, 6-0 prolene, backhand")
    assert r["ok"] and r["intent"] == "handoff" and "Castroviejo" in r["say"]
    wait_for(bridge, lambda: bridge.status.get("phase") == "presenting")
    assert status_of(bridge, "needle_holder_castroviejo") == HELD

    assert bridge.handle_text("got it")["intent"] == "release"
    wait_for(bridge, lambda: status_of(bridge, "needle_holder_castroviejo") == IN_FIELD)

    c = bridge.count_check()                                    # the surgeon still has it
    assert not c["ok"] and c["outstanding"] == ["Castroviejo needle holder"] and c["source"] == "simulated"

    assert bridge.handle_text("I'm done with the castro")["intent"] == "used"
    assert status_of(bridge, "needle_holder_castroviejo") == USED
    assert bridge.count_check()["ok"]
    assert {"utterance", "intent", "command", "job", "ledger", "count", "say"} <= set(kinds(bridge))


def test_the_demo_moment_a_missing_instrument_is_caught_and_named(rig):
    bridge, _ = rig
    assert bridge.count_check()["ok"]
    assert bridge.sim_remove("scalpel_11")
    r = bridge.handle_text("count check")
    assert r["intent"] == "count" and "Missing: #11 scalpel" in r["say"]
    bridge.sim_restore()
    assert bridge.count_check()["ok"]


def test_a_failed_pick_puts_the_instrument_back_on_the_tray(rig):
    bridge, server = rig
    server.app.driver.bus.object_in_jaws = False
    bridge.handle_text("potts")
    wait_for(bridge, lambda: "job" in kinds(bridge))
    assert status_of(bridge, "scissors_potts") == TRAY
    job = [e for e in bridge.stream.events if e["kind"] == "job"][-1]["data"]
    assert job["status"] == "failed" and "missed the grasp" in job["error"]


def test_a_request_the_arm_refuses_is_spoken_not_raised(rig):
    bridge, _ = rig
    r = bridge.handle_text("hand me a needle holder with vicryl")
    assert r["intent"] == "ask" and "vicryl" in r["say"].lower()      # not silently a prolene one


def test_ambiguity_is_a_question_and_nothing_moves(rig):
    bridge, server = rig
    r = bridge.handle_text("hand me a banana")
    assert r["intent"] == "ask" and server.app.assistant.status()["last_job"] is None


def test_stop_and_reset_reach_the_arm(rig):
    bridge, server = rig
    assert bridge.handle_text("stop")["intent"] == "stop"
    assert bridge.handle_text("reset")["intent"] == "reset"


def test_every_step_lands_in_both_audit_logs_and_the_heads_are_anchored(rig):
    bridge, server = rig
    bridge.handle_text("potts")
    wait_for(bridge, lambda: bridge.status.get("phase") == "presenting")
    bridge.handle_text("got it")
    wait_for(bridge, lambda: "job" in kinds(bridge))
    state = bridge.anchor()
    assert state["console"]["ok"] and state["arm"]["ok"] and state["arm"]["entries"] >= 2
    anchors = [e["data"] for e in bridge.stream.events if e["kind"] == "audit"]
    assert {a["source"] for a in anchors} == {"console", "arm"} and all(a["head"] for a in anchors)
    kinds_logged = [__import__("json").loads(l)["kind"] for l in bridge.audit.path.read_text().splitlines()]
    assert {"utterance", "command", "job"} <= set(kinds_logged)


def test_sim_controls_are_refused_outside_simulation(arm_server):
    arm = ArmClient(arm_server.url, arm_server.token)
    b = Bridge(arm, Ledger.from_tray(load_tray_config()), lambda t, c: None, sim=False)
    with pytest.raises(PermissionError):
        b.sim_remove("scalpel_11")


class ScriptedArm:
    """Stands in for ArmClient: status() replays a script, then repeats the last entry."""

    def __init__(self, *statuses):
        self.script, self.i = list(statuses), 0

    def status(self):
        s = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        if isinstance(s, Exception):
            raise s
        return s

    def audit(self):
        return {"ok": True, "entries": 0, "head": "0" * 64}


def idle(fault=None, **kw):
    return {"phase": "fault" if fault else "idle", "holding": None, "job": None, "last_job": None, "events": [],
            "arm": {"enabled": True, "fault": fault}, **kw}


def test_a_fault_triggers_triage_once_and_clears(arm_server):
    class Triage:
        def __init__(self):
            self.calls = []

        def trigger(self, fault, ctx):
            self.calls.append(fault)

        def snapshot(self):
            return []

    t = Triage()
    f = "joint3 is 28.0 deg off its setpoint (collision or blocked?) - arm frozen in place"
    b = Bridge(ScriptedArm(idle(), idle(f), idle(f), idle(), idle(f)), Ledger(["scalpel_11"]), lambda t_, c: None, triage=t)
    for _ in range(5):
        b.poll()
    assert t.calls == [f, f]                       # once per fault episode, not once per poll
    assert "fault" in kinds(b)


def test_arm_offline_is_reported_once_then_recovery_is_noticed():
    down = ArmError("arm unreachable")
    b = Bridge(ScriptedArm(down, down, idle()), Ledger(["scalpel_11"]), lambda t, c: None)
    for _ in range(3):
        b.poll()
    assert kinds(b).count("arm_offline") == 1 and "arm_online" in kinds(b) and b.online is True
