import json
import threading
import urllib.error
import urllib.request

import pytest

from scrub_arm.audit import AuditLog
from scrub_arm.server import serve

TOKEN = "s3cret-token"
GOOD_ORIGIN = "http://localhost:5173"


@pytest.fixture
def make_server(app, tmp_path):
    servers = []

    def _make(**kw):
        audit = AuditLog(tmp_path / "audit.jsonl")
        httpd = serve(app.assistant, app.catalog, "127.0.0.1", 0, audit=audit, **kw)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return f"http://127.0.0.1:{httpd.server_address[1]}", audit

    yield _make
    for s in servers:
        s.shutdown()
        s.server_close()


def call(base, path, body=None, headers=None, method=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read() or b"{}"), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}"), e.headers


def auth(token=TOKEN):
    return {"Authorization": f"Bearer {token}"}


def test_without_a_token_configured_plain_clients_still_work(make_server):
    base, _ = make_server()
    assert call(base, "/status")[0] == 200


def test_token_is_required_and_checked(make_server):
    base, audit = make_server(token=TOKEN)
    assert call(base, "/status")[0] == 401
    assert call(base, "/status", headers=auth("wrong"))[0] == 401
    assert call(base, "/status", headers=auth())[0] == 200
    assert call(base, "/home", {}, headers={})[0] == 401
    assert [json.loads(l)["kind"] for l in audit.path.read_text().splitlines()].count("refused") == 3


def test_no_wildcard_cors_header_is_ever_sent(make_server):
    base, _ = make_server(token=TOKEN)
    _, _, headers = call(base, "/status", headers=auth())
    assert headers.get("Access-Control-Allow-Origin") is None


def test_web_page_from_another_origin_cannot_command_the_arm(make_server, app):
    base, audit = make_server()      # even with no token at all
    evil = {"Origin": "https://evil.example"}
    code, body, headers = call(base, "/handoff", {"instrument": "resano"}, headers=evil)
    assert code == 403 and headers.get("Access-Control-Allow-Origin") is None
    assert app.assistant.status()["job"] is None and app.assistant.status()["last_job"] is None
    assert call(base, "/status", headers=evil)[0] == 403


def test_allowed_origin_gets_cors_and_can_call_with_the_token(make_server):
    base, _ = make_server(token=TOKEN, allowed_origins=[GOOD_ORIGIN])
    code, _, headers = call(base, "/status", headers={**auth(), "Origin": GOOD_ORIGIN})
    assert code == 200 and headers["Access-Control-Allow-Origin"] == GOOD_ORIGIN
    # the browser's preflight carries no token, so it must be answered without one
    code, _, headers = call(base, "/handoff", method="OPTIONS", headers={"Origin": GOOD_ORIGIN})
    assert code == 204 and headers["Access-Control-Allow-Origin"] == GOOD_ORIGIN
    assert "Authorization" in headers["Access-Control-Allow-Headers"]


def test_preflight_from_a_stranger_gets_no_cors_headers(make_server):
    base, _ = make_server(allowed_origins=[GOOD_ORIGIN])
    _, _, headers = call(base, "/handoff", method="OPTIONS", headers={"Origin": "https://evil.example"})
    assert headers.get("Access-Control-Allow-Origin") is None


def test_dns_rebinding_host_header_is_refused_on_loopback(make_server):
    base, _ = make_server()
    assert call(base, "/status", headers={"Host": "evil.example"})[0] == 403
    assert call(base, "/status", headers={"Host": "localhost:8765"})[0] == 200


def test_requests_are_audited_with_their_fields_and_status(make_server):
    base, audit = make_server(token=TOKEN)
    assert call(base, "/resolve", {"instrument": "potts", "orientation": "backhand"}, headers=auth())[0] == 200
    assert call(base, "/handoff", {"instrument": "hammer"}, headers=auth())[0] == 400
    entries = [json.loads(l) for l in audit.path.read_text().splitlines()]
    resolve, bad = entries
    assert resolve["data"]["path"] == "/resolve" and resolve["data"]["status"] == 200
    assert resolve["data"]["instrument"] == "potts" and resolve["data"]["orientation"] == "backhand"
    assert bad["data"]["status"] == 400
    assert TOKEN not in audit.path.read_text()


def test_audit_endpoint_reports_a_tampered_log(make_server):
    base, audit = make_server(token=TOKEN)
    call(base, "/resolve", {"instrument": "potts"}, headers=auth())
    code, body, _ = call(base, "/audit", headers=auth())
    assert code == 200 and body["ok"] and body["entries"] == 1
    audit.path.write_text(audit.path.read_text().replace("potts", "ryder"))
    assert call(base, "/audit", headers=auth())[1]["ok"] is False


def test_status_since_returns_samples_and_still_needs_the_token(make_server):
    base, _ = make_server(token=TOKEN)
    assert call(base, "/status?since=0")[0] == 401
    code, body, _ = call(base, "/status?since=0", headers=auth())
    assert code == 200 and isinstance(body["samples"], list)
    assert "samples" not in call(base, "/status", headers=auth())[1]
    for bad in ("abc", "nan", "inf"):
        assert call(base, f"/status?since={bad}", headers=auth())[0] == 400
