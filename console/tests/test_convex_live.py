"""Opt-in integration test against a real Convex deployment (e.g. `CONVEX_AGENT_MODE=anonymous npx convex dev`).

    CONVEX_LIVE_URL=http://127.0.0.1:3210 CONVEX_LIVE_KEY=<admin key> pytest tests/test_convex_live.py

For the anonymous local deployment the admin key is `adminKey` in console/.convex/local/default/config.json.
"""
import os
import time
import uuid

import pytest

pytest.importorskip("convex")
from convex import ConvexClient, ConvexError  # noqa: E402

from scrub_console.sinks import ConvexSink  # noqa: E402

URL, KEY = os.environ.get("CONVEX_LIVE_URL"), os.environ.get("CONVEX_LIVE_KEY")
pytestmark = pytest.mark.skipif(not (URL and KEY), reason="set CONVEX_LIVE_URL and CONVEX_LIVE_KEY to run")


@pytest.fixture
def sink():
    return ConvexSink(URL, KEY)


def test_sink_writes_land_and_public_live_queries_read_them(sink):
    tag = uuid.uuid4().hex[:8]
    sink.publish("utterance", {"text": f"hello {tag}"})
    sink.publish("ledger", {"items": [{"uid": f"x{tag}#1", "instrument": "forceps_resano", "name": "Resano forceps",
                                       "status": "in_field"}], "counts": {"in_field": 1}})
    sink.publish("audit", {"source": f"arm-{tag}", "seq": 3, "head": "abc", "ok": True})
    sink.publish("proposal", {"id": f"t{tag}", "fault": "joint3 lag", "status": "pending", "created": time.time(),
                              "simulated": True, "summary": "s", "diff": "+x", "changes": [{"key": "k", "old": 1, "new": 2}],
                              "errors": [], "steps": [{"type": "tool", "name": "Read"}], "needs_restart": False,
                              "original_sha": "00"})
    assert sink.flush(10) and sink.failures == 0

    public = ConvexClient(URL)                                  # no credentials: what a browser gets
    assert any(e["data"].get("text") == f"hello {tag}" for e in public.query("events:recent"))
    assert any(i["uid"] == f"x{tag}#1" for i in public.query("instruments:list")["items"])
    assert public.query("audit:latest")[f"arm-{tag}"]["seq"] == 3
    assert any(p["id"] == f"t{tag}" and p["status"] == "pending" for p in public.query("proposals:recent"))


def test_the_public_url_cannot_forge_history():
    public = ConvexClient(URL)
    with pytest.raises(Exception):
        public.mutation("events:add", {"kind": "count", "data": {"ok": True}, "seq": 1, "t": 1.0})
    with pytest.raises(Exception):
        public.mutation("audit:anchor", {"source": "arm", "seq": 1, "head": "x", "ok": True})


def test_a_log_that_moves_backwards_raises_an_alert(sink):
    src = f"arm-{uuid.uuid4().hex[:8]}"
    sink.publish("audit", {"source": src, "seq": 9, "head": "h9", "ok": True})
    sink.publish("audit", {"source": src, "seq": 4, "head": "h4", "ok": True})
    assert sink.flush(10)
    alerts = ConvexClient(URL).query("audit:alerts")
    assert any(a["source"] == src and a["kind"] == "rollback" for a in alerts)
