import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scrub_arm.app import build
from scrub_arm.perception_client import PerceptionClient

TARGET = {"id": "forceps_resano", "x_mm": 300, "y_mm": 0, "yaw_deg": 0, "thickness_mm": 4,
          "confidence": 0.9, "exposure": 0.95}
OTHER = {**TARGET, "id": "scissors_potts", "x_mm": 310, "y_mm": 30}
GRASP = {"action": "grasp_target", "target": TARGET, "place": None, "say": "Resano"}
ASIDE = {"action": "move_aside", "target": OTHER, "place": {"x_mm": 250, "y_mm": -200, "yaw_deg": 0}, "say": ""}
FAIL = {"action": "fail", "target": None, "place": None, "say": "I don't see a Resano forceps"}


class FakePerception:
    def __init__(self, script):
        self.script, self.calls = list(script), 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                body = outer.script[min(outer.calls, len(outer.script) - 1)]
                outer.calls += 1
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()


@pytest.fixture
def make_app(config_dir):
    apps, servers = [], []

    def _make(script=None, url=None):
        fake = FakePerception(script) if script is not None else None
        p = config_dir / "arm.yaml"
        t = re.sub(r"^(\s*mode:\s*)slots", r"\g<1>ik", p.read_text(), flags=re.M)
        t = re.sub(r"^(\s*perception_url:\s*)\S+", rf"\g<1>{url or fake.url}", t, flags=re.M)
        p.write_text(t)
        a = build(sim=True, config_dir=config_dir)
        m = a.cfg.motion
        m.speed_scale, m.approach_speed_scale, m.min_move_s, m.settle_timeout_s = 20.0, 1.0, 0.02, 0.1
        a.cfg.gripper.settle_s = 0.05
        a.cfg.handoff.release_wait_s = 0.02
        a.cfg.handoff.present_timeout_s = 0.3
        a.driver.start()
        a.driver.enable()
        apps.append(a)
        if fake:
            servers.append(fake)
        return a, fake

    yield _make
    for a in apps:
        a.driver.shutdown()
    for f in servers:
        f.httpd.shutdown()


def run(app, text="resano"):
    r = app.catalog.resolve(text)
    app.assistant.submit("handoff", r)
    return app.assistant.wait_idle(timeout=30)


def test_grasp_target_first_try(make_app):
    app, fake = make_app([GRASP])
    job = run(app)
    assert job["status"] == "done", job
    assert fake.calls == 1


def test_move_aside_then_grasp(make_app):
    app, fake = make_app([ASIDE, GRASP])
    job = run(app)
    assert job["status"] == "done", job
    assert fake.calls == 2


def test_fail_action_speaks(make_app):
    app, fake = make_app([FAIL])
    job = run(app)
    assert job["status"] == "failed"
    assert "see" in job["error"]


def test_perception_down_fails_cleanly(make_app):
    app, _ = make_app(url="http://127.0.0.1:9")
    job = run(app)
    assert job["status"] == "failed"
    assert "perception" in job["error"]


def test_malformed_plan_fails_cleanly(make_app):
    app, _ = make_app([b"not json"])
    job = run(app)
    assert job["status"] == "failed" and "perception" in job["error"]


def test_cap_at_max_moves(make_app):
    app, fake = make_app([ASIDE] * 10)
    job = run(app)
    assert job["status"] == "failed"
    assert fake.calls == app.cfg.pick.max_declutter_moves


def test_missed_grasp_replans_once(make_app):
    app, fake = make_app([GRASP])
    app.driver.bus.object_in_jaws = False
    job = run(app)
    assert job["status"] == "failed"
    assert fake.calls == 2


def test_client_rejects_missing_action():
    fake = FakePerception([{"say": "hi"}])
    from scrub_arm.handoff import TaskFailed
    with pytest.raises(TaskFailed):
        PerceptionClient(fake.url).plan("forceps_resano")
    fake.httpd.shutdown()


def test_unreachable_place_fails_before_lifting_the_mover(make_app):
    far = {**ASIDE, "place": {"x_mm": 900, "y_mm": 0, "yaw_deg": 0}}
    app, fake = make_app([far])
    job = run(app)
    assert job["status"] == "failed"
    assert app.assistant.holding is None
    assert not any("plan: move_aside" in e["msg"] and "picked" in e["msg"] for e in app.assistant.events)
    assert app.assistant._picked_from is None          # never picked anything up


def test_stop_while_carrying_a_mover_leaves_it_held_not_forgotten(make_app):
    app, fake = make_app([ASIDE, GRASP])
    real_place = app.assistant._place_at

    def stop_instead(*a, **k):
        app.assistant.stop()
        app.driver.sleep(0.01)        # raises MotionAborted
    app.assistant._place_at = stop_instead
    job = run(app)
    assert job["status"] == "stopped"
    assert app.assistant.holding is not None and app.assistant.holding.instrument.id == "scissors_potts"
    app.assistant._place_at = real_place


@pytest.mark.parametrize("bad", [{"action": "grasp_target", "target": {}, "place": None, "say": ""},
                                 {"action": "grasp_target", "target": {**TARGET, "x_mm": "nan"}, "place": None, "say": ""},
                                 {"action": "grasp_target", "target": {**TARGET, "x_mm": None}, "place": None, "say": ""}])
def test_bad_numbers_fail_with_speech_not_fault(bad):
    fake = FakePerception([bad])
    from scrub_arm.handoff import TaskFailed
    with pytest.raises(TaskFailed):
        PerceptionClient(fake.url).plan("forceps_resano")
    fake.httpd.shutdown()
