import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from scrub_arm.server import serve


@pytest.fixture
def base_url(app):
    httpd = serve(app.assistant, app.catalog, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_resolve_is_a_dry_run(base_url):
    code, body = call(base_url, "/resolve", {"instrument": "needle holder", "suture": "prolene", "size": "6-0",
                                             "orientation": "backhand"})
    assert code == 200
    assert body["slot"] == "A1" and body["orientation"] == "backhand"
    assert body["say"] == "Castroviejo needle holder, 6-0, Prolene, backhand"


def test_handoff_over_http(base_url, app):
    code, body = call(base_url, "/handoff", {"text": "needle holder, 4-0 prolene, backhand"})
    assert code == 202 and body["slot"] == "A2"
    code, _ = call(base_url, "/home", {})
    assert code == 409
    for _ in range(500):
        _, st = call(base_url, "/status")
        if st["phase"] == "presenting":
            break
        time.sleep(0.01)
    assert st["holding_slot"] == "A2"
    assert call(base_url, "/release", {})[0] == 200
    app.assistant.wait_idle()
    _, st = call(base_url, "/status")
    assert st["phase"] == "idle" and st["last_job"]["status"] == "done"


def test_bad_requests_return_speakable_errors(base_url):
    code, body = call(base_url, "/handoff", {"instrument": "needle holder", "suture": "vicryl"})
    assert code == 400 and "vicryl" in body["say"]
    code, body = call(base_url, "/handoff", {"instrument": ""})
    assert code == 400
