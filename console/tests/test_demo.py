import json
import urllib.request

import pytest

from conftest import wait_for
from scrub_console.config import Settings
from scrub_console.demo import start


@pytest.fixture
def rig(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    d = start(Settings(audit_dir=tmp_path), fast=True)
    yield d
    d.stop()


def post(rig, path, body):
    req = urllib.request.Request(rig.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_the_rehearsal_rig_runs_a_whole_case_with_no_hardware(rig):
    b = rig.console.bridge
    assert rig.console.capabilities()["sim"] and not rig.console.capabilities()["claude"]
    assert post(rig, "/api/utterance", {"text": "needle holder, 4-0 prolene"})["ok"]
    wait_for(b, lambda: b.status.get("phase") == "presenting")
    post(rig, "/api/utterance", {"text": "got it"})
    wait_for(b, lambda: b.ledger.counts()["in_field"] == 1)
    post(rig, "/api/utterance", {"text": "done with the ryder"})
    assert post(rig, "/api/count", {})["ok"]
    post(rig, "/api/sim/remove", {"instrument": "forceps_debakey"})
    r = post(rig, "/api/count", {})
    assert not r["ok"] and r["missing"] == ["DeBakey forceps"]
