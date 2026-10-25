import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scrub_console.sinks import ConvexHTTP, ConvexSink


@pytest.fixture
def fake_convex():
    """Answers POST /api/mutation like Convex: {"status": "success", "value": ...} or {"status": "error", ...}."""
    got = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            got.append((self.path, self.headers.get("Authorization"), body))
            reply = ({"status": "error", "errorMessage": "Could not find public function"} if body["path"] == "nope:x"
                     else {"status": "success", "value": {"echo": body["args"]}})
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/", got
    httpd.shutdown()


def test_mutations_go_over_http_with_admin_auth_and_errors_raise(fake_convex):
    url, got = fake_convex
    c = ConvexHTTP(url, "dev:abc|secret", timeout_s=5)
    assert c.mutation("metrics:record", {"t0": 1.5, "samples": [{"q": [None, 2.0]}]}) == {
        "echo": {"t0": 1.5, "samples": [{"q": [None, 2.0]}]}}
    path, auth, body = got[0]
    assert path == "/api/mutation" and auth == "Convex dev:abc|secret"
    assert body == {"path": "metrics:record", "args": {"t0": 1.5, "samples": [{"q": [None, 2.0]}]}, "format": "json"}
    with pytest.raises(RuntimeError, match="nope:x: Could not find"):
        c.mutation("nope:x", {})
    ConvexHTTP(url).mutation("events:recent", {})
    assert got[-1][1] is None                                 # no key, no Authorization header


def test_the_sink_routes_metrics_and_survives_a_failing_write(fake_convex):
    url, got = fake_convex
    sink = ConvexSink(url, "k")
    sink.publish("metrics", {"t0": 1.0, "t1": 6.0, "source": "sim", "samples": []})
    assert sink.flush(10)
    sink.client.url = "http://127.0.0.1:9"                     # nothing listens: the write fails, the sink lives on
    sink.publish("utterance", {"text": "hi"})
    assert sink.flush(10)
    sink.client.url = url.rstrip("/")
    sink.publish("utterance", {"text": "again"})
    assert sink.flush(10)
    assert [b["path"] for _, _, b in got] == ["metrics:record", "events:add"] and sink.failures == 1
