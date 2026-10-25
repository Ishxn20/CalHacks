import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
from conftest import wait_for

from scrub_console.app import build_console
from scrub_console.config import Settings
from scrub_console.server import serve
from scrub_console.speech import CartesiaSpeech


class FakeCartesia:
    def __init__(self):
        self.stt = SimpleNamespace(transcribe=lambda **kw: SimpleNamespace(text="castroviejo backhand"))
        self.tts = SimpleNamespace(generate=lambda **kw: SimpleNamespace(read=lambda: b"RIFFwave"))
        self.voices = SimpleNamespace(list=lambda limit=None: [SimpleNamespace(id="v", name="n")])


@pytest.fixture
def console(arm_server, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    s = Settings(arm_url=arm_server.url, arm_token=arm_server.token, audit_dir=tmp_path / "logs")
    return build_console(s, sim=True, demo=True, rehearse_droid=True, arm_config=arm_server.config_path)


@pytest.fixture
def site(console):
    httpd = serve(console, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield SimpleNamespace(base=base, console=console, port=httpd.server_address[1])
    httpd.shutdown()
    httpd.server_close()


def call(site, path, body=None, headers=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(site.base + path, data=data, headers=headers or {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            ctype = r.headers.get("Content-Type", "")
            payload = r.read()
            return r.status, (json.loads(payload) if "json" in ctype else payload), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), e.headers


def test_the_dashboard_and_state_are_served(site):
    code, html, h = call(site, "/")
    assert code == 200 and b"Scrub console" in html and "text/html" in h["Content-Type"]
    assert b"Doctor view" in html and b"Robot view" in html
    code, camera_js, h = call(site, "/camera.js")
    assert code == 200 and b"DemoCameraSource" in camera_js and "text/javascript" in h["Content-Type"]
    code, st, _ = call(site, "/api/state")
    assert code == 200 and st["sim"] is True and len(st["ledger"]["items"]) == 8
    assert st["capabilities"]["claude"] is False and st["capabilities"]["droid"] == "rehearsal"


def test_a_typed_request_reaches_the_arm_and_the_board_follows(site):
    b = site.console.bridge
    b.poll()
    code, r, _ = call(site, "/api/utterance", {"text": "castroviejo, backhand"})
    assert code == 200 and r["ok"] and r["intent"] == "handoff"
    wait_for(b, lambda: b.status.get("phase") == "presenting")
    call(site, "/api/utterance", {"text": "got it"})
    wait_for(b, lambda: any(i["status"] == "in_field" for i in b.ledger.snapshot()))
    _, st, _ = call(site, "/api/state")
    assert st["ledger"]["counts"]["in_field"] == 1


def test_other_websites_cannot_drive_the_console(site):
    assert call(site, "/api/state", headers={"Origin": "https://evil.example"})[0] == 403
    assert call(site, "/api/utterance", {"text": "stop"}, headers={"Origin": "https://evil.example", "Content-Type": "application/json"})[0] == 403
    assert call(site, "/api/state", headers={"Host": "evil.example"})[0] == 403
    ok = call(site, "/api/state", headers={"Origin": f"http://127.0.0.1:{site.port}"})
    assert ok[0] == 200 and "Access-Control-Allow-Origin" not in ok[2]


def test_the_console_refuses_to_listen_off_loopback(console):
    with pytest.raises(ValueError):
        serve(console, "0.0.0.0", 0)


def test_events_stream_live_over_sse(site):
    site.console.bridge.emit("warning", message="hello stream")
    conn = http.client.HTTPConnection("127.0.0.1", site.port, timeout=5)
    conn.request("GET", "/api/events?since=0")
    resp = conn.getresponse()
    assert resp.status == 200 and resp.getheader("Content-Type") == "text/event-stream"
    buf = b""
    deadline = time.time() + 5
    while b"hello stream" not in buf and time.time() < deadline:
        buf += resp.fp.readline()
    conn.close()
    assert b"data:" in buf and b"hello stream" in buf


def test_voice_in_and_speech_out_use_cartesia(site):
    fake = FakeCartesia()
    site.console.speech = CartesiaSpeech(client=fake, voice_id="v")
    code, r, _ = call(site, "/api/audio", raw=b"RIFFfake", headers={"Content-Type": "audio/wav"})
    assert code == 200 and r == {"ignored": True}               # no wake word: OR talk, dropped
    fake.stt.transcribe = lambda **kw: SimpleNamespace(text="Scrub, castroviejo backhand")
    code, r, _ = call(site, "/api/audio", raw=b"RIFFfake", headers={"Content-Type": "audio/wav"})
    assert code == 200 and r["heard"] == "castroviejo backhand" and r["intent"] == "handoff"
    code, audio, h = call(site, "/api/tts?text=Releasing.", headers={})
    assert code == 200 and audio == b"RIFFwave" and h["Content-Type"] == "audio/wav"


def test_without_a_cartesia_key_voice_says_so_and_typing_still_works(site):
    code, r, _ = call(site, "/api/tts?text=hello", headers={})
    assert code == 503 and "CARTESIA_API_KEY" in r["error"]
    code, r, _ = call(site, "/api/audio", raw=b"x", headers={"Content-Type": "audio/wav"})
    assert code == 503
    assert call(site, "/api/utterance", {"text": "stop"})[0] == 200


def test_browser_speech_is_gated_by_the_wake_word_and_chatter_leaves_no_trace(site):
    assert call(site, "/api/heard", {"text": "can someone get the lights"})[1] == {"ignored": True}
    assert call(site, "/api/heard", {"text": "Scrub"})[1]["say"] == "Yes?"
    assert call(site, "/api/heard", {"text": "count check"})[1]["intent"] == "count"      # follow-up, no wake word
    said = [e["data"]["text"] for e in site.console.bridge.stream.of_kind("utterance")]
    assert said == ["count check"]                                                     # the chatter was never recorded


def test_the_count_check_catches_what_was_removed(site):
    assert call(site, "/api/count", {})[1]["ok"] is True
    assert call(site, "/api/sim/remove", {"instrument": "scalpel_11"})[1]["removed"] is True
    r = call(site, "/api/count", {})[1]
    assert not r["ok"] and r["missing"] == ["#11 scalpel"] and r["source"] == "simulated"
    call(site, "/api/sim/restore", {})
    assert call(site, "/api/count", {})[1]["ok"] is True


def test_report_and_prediction_work_without_keys_and_say_how(site):
    call(site, "/api/utterance", {"text": "count check"})
    code, r, _ = call(site, "/api/report", {})
    assert code == 200 and r["generated_by"] == "template" and "Last count" in r["text"]
    r = call(site, "/api/predict", {})[1]
    assert r == {"prediction": None, "available": False}


def test_audit_chains_and_the_non_destructive_tamper_demo(site):
    call(site, "/api/utterance", {"text": "stop"})
    a = call(site, "/api/audit")[1]
    assert a["console"]["ok"] and a["arm"]["ok"]
    t = call(site, "/api/demo/tamper", {})[1]
    assert t["detected"] is True and "line" in t["error"] and t["original_intact"] is True
    assert site.console.audit.verify().ok                         # the real log was never touched


def test_a_fault_gets_a_proposal_a_human_approves(site, arm_server):
    before = arm_server.config_path.read_text()
    code, r, _ = call(site, "/api/triage/trigger", {})
    assert code == 200
    b = site.console.bridge
    deadline = time.time() + 5
    while time.time() < deadline and site.console.triage.proposals[r["id"]].status == "running":
        time.sleep(0.05)
    p = site.console.triage.proposals[r["id"]]
    assert p.status == "pending" and p.simulated and arm_server.config_path.read_text() == before
    code, out, _ = call(site, "/api/triage/approve", {"id": r["id"]})
    assert code == 200 and out["status"] == "approved" and arm_server.config_path.read_text() != before
    assert call(site, "/api/triage/approve", {"id": r["id"]})[0] == 400          # a decision is final


def test_demo_and_simulation_tools_are_off_when_not_enabled(arm_server, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = Settings(arm_url=arm_server.url, arm_token=arm_server.token, audit_dir=tmp_path / "logs2")
    c = build_console(s, sim=False, demo=False)
    httpd = serve(c, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    site = SimpleNamespace(base=f"http://127.0.0.1:{httpd.server_address[1]}")
    try:
        assert call(site, "/api/demo/tamper", {})[0] == 404
        assert call(site, "/api/triage/trigger", {})[0] == 404
        assert call(site, "/api/sim/remove", {"instrument": "scalpel_11"})[0] == 403
    finally:
        httpd.shutdown()
        httpd.server_close()
