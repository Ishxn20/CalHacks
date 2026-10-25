import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scrub_console.report import BitdeerClient, BitdeerError, build_facts, predict_next, summarize_case, template_report


class FakeBitdeer:
    def __init__(self, reply="All good.", status=200):
        self.requests, outer = [], self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, self.headers["Authorization"], body))
                data = json.dumps({"choices": [{"message": {"role": "assistant", "content": reply}}]}).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def stop(self):
        self.httpd.shutdown()


@pytest.fixture
def bitdeer():
    made = []

    def make(**kw):
        f = FakeBitdeer(**kw)
        made.append(f)
        return f, BitdeerClient("key123", f.url, "some/model", timeout_s=5)

    yield make
    for f in made:
        f.stop()


EVENTS = [
    {"kind": "utterance", "data": {"text": "resano"}},
    {"kind": "job", "data": {"kind": "handoff", "status": "done", "result": "handed"}},
    {"kind": "job", "data": {"kind": "handoff", "status": "failed", "error": "missed the grasp"}},
    {"kind": "fault", "data": {"message": "joint3 lag"}},
    {"kind": "proposal", "data": {"status": "pending", "simulated": True, "changes": []}},
    {"kind": "count", "data": {"ok": False, "missing": ["#11 scalpel"], "outstanding": [], "unexpected": [],
                                "source": "simulated", "say": "Count is off. Missing: #11 scalpel."}},
]
AUDIT = {"arm": {"ok": True, "entries": 12}, "console": {"ok": False, "entries": 30}}


def test_facts_come_only_from_recorded_events():
    f = build_facts(EVENTS, {"tray": 7}, AUDIT)
    assert f["requests"] == 1 and f["instrument_jobs"] == 2 and f["completed"] == 1
    assert f["failed"] == [{"kind": "handoff", "error": "missed the grasp"}]
    assert f["last_count"]["missing"] == ["#11 scalpel"] and f["faults"] == ["joint3 lag"]


def test_template_report_leads_with_the_discrepancy_and_flags_a_bad_audit_log():
    text = template_report(build_facts(EVENTS, {}, AUDIT))
    assert "DISCREPANCY" in text and "#11 scalpel" in text and "NOT INTACT" in text and "rehearsal" in text


def test_bitdeer_is_called_the_openai_compatible_way(bitdeer):
    server, client = bitdeer(reply="Count was off: the #11 scalpel is missing.")
    out = summarize_case(build_facts(EVENTS, {}, AUDIT), client)
    path, auth, body = server.requests[0]
    assert path == "/v1/chat/completions" and auth == "Bearer key123" and body["model"] == "some/model"
    assert "Use ONLY the JSON facts" in body["messages"][0]["content"] and "missed the grasp" in body["messages"][0]["content"]
    assert out == {"text": "Count was off: the #11 scalpel is missing.", "generated_by": "bitdeer:some/model"}


def test_report_falls_back_to_the_template_when_bitdeer_is_down():
    client = BitdeerClient("k", "http://127.0.0.1:9/v1", timeout_s=1)
    out = summarize_case(build_facts(EVENTS, {}, AUDIT), client)
    assert out["generated_by"] == "template" and "DISCREPANCY" in out["text"]
    assert summarize_case(build_facts(EVENTS, {}, AUDIT))["generated_by"] == "template"


def test_client_raises_a_clear_error():
    with pytest.raises(BitdeerError):
        BitdeerClient("k", "http://127.0.0.1:9/v1", timeout_s=1).chat([{"role": "user", "content": "hi"}])


REMAINING = {"scalpel_11": "#11 scalpel", "forceps_resano": "Resano forceps"}


def test_prediction_must_name_an_instrument_that_is_on_the_tray(bitdeer):
    _, ok = bitdeer(reply='Sure! {"instrument": "forceps_resano", "reason": "Forceps usually follow the scalpel."}')
    p = predict_next(["scalpel_11"], REMAINING, ok)
    assert p["instrument"] == "forceps_resano" and p["name"] == "Resano forceps" and p["generated_by"] == "bitdeer:some/model"
    _, bad = bitdeer(reply='{"instrument": "hammer", "reason": "x"}')
    assert predict_next([], REMAINING, bad) is None
    _, junk = bitdeer(reply="no idea")
    assert predict_next([], REMAINING, junk) is None


def test_no_prediction_without_a_model_or_candidates(bitdeer):
    _, client = bitdeer()
    assert predict_next([], REMAINING, None) is None and predict_next([], {}, client) is None
