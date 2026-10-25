import threading

import pytest
from conftest import wait_for

from scrub_console.app import build_console
from scrub_console.asi_agent import Console, agent_seed, respond
from scrub_console.config import Settings
from scrub_console.server import serve


@pytest.fixture
def chat(arm_server, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    s = Settings(arm_url=arm_server.url, arm_token=arm_server.token, audit_dir=tmp_path / "logs")
    console = build_console(s, sim=True, arm_config=arm_server.config_path)
    httpd = serve(console, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pending = {}
    say = lambda text, sender="me", allowed=frozenset(): respond(text, sender, pending, Console(  # noqa: E731
        f"http://127.0.0.1:{httpd.server_address[1]}"), set(allowed))
    yield say, console.bridge, pending
    httpd.shutdown()
    httpd.server_close()


def test_motion_is_read_back_and_waits_for_confirm(chat):
    say, bridge, pending = chat
    r = say("potts scissors")
    assert "Potts" in r and 'Reply "confirm"' in r and pending == {"me": "potts scissors"}
    assert not bridge.stream.of_kind("command")                   # nothing moved yet
    assert say("confirm").startswith("On it")
    wait_for(bridge, lambda: bridge.status.get("phase") == "presenting")


def test_cancel_moves_nothing_and_stop_is_immediate(chat):
    say, bridge, _ = chat
    say("potts scissors")
    assert say("cancel") == "Cancelled. Nothing moved."
    assert not bridge.stream.of_kind("command")
    assert say("stop").startswith("Stopped.")
    assert [e["data"]["name"] for e in bridge.stream.of_kind("command")] == ["stop"]


def test_status_and_count_are_answered_without_confirmation(chat):
    say, _, _ = chat
    r = say("what's with the surgeon?")
    assert "With the surgeon: none." in r and "Potts scissors" in r and "(simulated)" in r
    assert say("run the instrument count") == "Count correct. All 8 instruments are accounted for."


def test_a_real_arm_refuses_motion_from_unknown_senders(chat):
    say, bridge, pending = chat
    bridge.sim = False                                              # as if this were the real arm
    assert "only takes motion requests" in say("potts scissors", sender="stranger")
    assert pending == {}
    assert 'Reply "confirm"' in say("potts scissors", sender="team", allowed={"team"})


def test_the_seed_is_kept_so_the_agent_address_is_stable(tmp_path, monkeypatch):
    monkeypatch.delenv("SCRUB_AGENT_SEED", raising=False)
    first = agent_seed(tmp_path)
    assert first == agent_seed(tmp_path) and len(first) == 64
    assert oct((tmp_path / "asi_agent_seed").stat().st_mode)[-3:] == "600"
