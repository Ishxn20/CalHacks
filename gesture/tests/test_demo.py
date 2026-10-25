import base64
import json
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gesture.app import Controller, serve
from gesture.robot import Camera, Robot
from gesture.vision import ClaudeVision, GESTURES, VisionUnavailable, image_blocks

# A real 1x1 PNG; image transmission tests never contain camera footage.
PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="


class Clock:
    t = 100
    def __call__(self):
        return self.t


class FakeVision:
    def __init__(self):
        self.gesture, self.score, self.calls = "five_fingers", .96, 0
        self.left_hand_on_brown_table = True

    def classify(self, frames):
        self.calls += 1
        return {"gesture": self.gesture, "confidence": self.score, "observation": "Observed hand signal.",
                "source": "claude", "model": "fake", "left_hand_on_brown_table": self.left_hand_on_brown_table}


class FakeRobot:
    def __init__(self):
        self.jobs = []
        self.busy = self.stopped = False
        self.camera = SimpleNamespace(jpeg=lambda: None)

    def snapshot(self):
        return {"busy": self.busy, "stopped": self.stopped, "simulated": True, "phase": "idle", "message": "Ready.", "camera_error": "No camera."}

    def submit(self, tool):
        if self.busy or self.stopped:
            return False
        self.jobs.append(tool)
        return True

    def control(self, action):
        self.stopped = action == "stop"


@pytest.fixture
def rig():
    clock, vision, robot = Clock(), FakeVision(), FakeRobot()
    c = Controller(vision, robot, clock=clock)
    c.configure(True, True)
    return SimpleNamespace(c=c, clock=clock, vision=vision, robot=robot)


def tick(rig):
    rig.clock.t += 3
    return rig.c.analyze([PNG], rig.c.generation)


@pytest.mark.parametrize("gesture,tool", [(k,v["instrument"]) for k,v in GESTURES.items()])
def test_each_fixed_mapping_needs_two_readings_and_does_not_repeat(rig, gesture, tool):
    rig.vision.gesture = gesture
    assert tick(rig)["prediction"]["confirmed"] is False
    assert rig.robot.jobs == []
    assert tick(rig)["prediction"]["picked"] is True
    assert rig.robot.jobs == [tool]
    rig.clock.t += 30
    tick(rig)
    assert rig.robot.jobs == [tool]
    rig.vision.gesture = "none"
    tick(rig); tick(rig)
    rig.vision.gesture = gesture
    tick(rig); tick(rig)
    assert rig.robot.jobs == [tool,tool]


def test_low_confidence_and_different_gesture_break_confirmation(rig):
    tick(rig)
    rig.vision.score = .84
    assert tick(rig)["streak"] == 0
    rig.vision.score, rig.vision.gesture = .95, "four_fingers"
    assert tick(rig)["streak"] == 1
    rig.vision.gesture = "thumb_right_angle"
    assert tick(rig)["streak"] == 1
    assert rig.robot.jobs == []


def test_unknown_and_no_hands_never_pick(rig):
    for gesture in ["none", "invented_tool", "one_finger", "two_fingers", "pinch", "thumb_only", "index_only", "thumb_index"]:
        rig.vision.gesture = gesture
        tick(rig); tick(rig)
    assert not rig.robot.jobs


@pytest.mark.parametrize("gesture", list(GESTURES))
def test_every_pose_requires_the_left_hand_on_the_brown_table(rig,gesture):
    rig.vision.gesture=gesture
    rig.vision.left_hand_on_brown_table=False
    rig.vision.score=.99
    tick(rig);state=tick(rig)
    assert state["prediction"]["gesture"]=="none"
    assert not state["prediction"]["confirmed"] and not rig.robot.jobs
    rig.vision.left_hand_on_brown_table=True
    tick(rig);tick(rig)
    assert rig.robot.jobs==[GESTURES[gesture]["instrument"]]


def test_fast_mode_uses_one_high_confidence_sequence_and_keeps_duplicate_guard():
    clock,vision,robot=Clock(),FakeVision(),FakeRobot()
    c=Controller(vision,robot,clock=clock,confirmations=1,threshold=.9)
    c.configure(True,True)
    vision.score=.89
    assert not c.analyze([PNG]*3,c.generation)["prediction"]["confirmed"]
    assert not robot.jobs
    clock.t+=1;vision.score=.95
    state=c.analyze([PNG]*3,c.generation)
    assert state["prediction"]["confirmed"] and state["prediction"]["picked"]
    assert robot.jobs==["scalpel_10"]
    clock.t+=20;c.analyze([PNG]*3,c.generation)
    assert robot.jobs==["scalpel_10"]


def test_busy_and_cooldown_block_requests(rig):
    rig.robot.busy = True
    tick(rig); tick(rig)
    assert not rig.robot.jobs
    rig.robot.busy = False
    tick(rig)
    assert len(rig.robot.jobs) == 1
    rig.vision.gesture = "four_fingers"
    tick(rig); tick(rig)
    assert len(rig.robot.jobs) == 1
    rig.clock.t += 10
    tick(rig)
    assert len(rig.robot.jobs) == 2


def test_manual_pick_requires_enabled_current_fresh_confirmed_session(rig):
    rig.c.configure(True, False)
    with pytest.raises(ValueError):
        rig.c.prepare(rig.c.generation)
    tick(rig); tick(rig)
    assert not rig.robot.jobs
    with pytest.raises(ValueError):
        rig.c.prepare(rig.c.generation-1)
    rig.clock.t += 11
    with pytest.raises(ValueError, match="stale"):
        rig.c.prepare(rig.c.generation)
    tick(rig)
    rig.c.prepare(rig.c.generation)
    assert rig.robot.jobs == ["scalpel_10"]
    rig.c.configure(False, False)
    with pytest.raises(ValueError):
        rig.c.prepare(rig.c.generation)


def test_model_failure_clears_confirmation(rig):
    tick(rig)
    original = rig.vision.classify
    rig.vision.classify = Mock(side_effect=VisionUnavailable("no model"))
    with pytest.raises(VisionUnavailable):
        tick(rig)
    assert rig.c.streak == 0 and rig.c.result is None
    rig.vision.classify = original
    tick(rig)
    assert not rig.robot.jobs


def test_stop_during_model_call_cannot_move_arm_and_only_one_call_runs(rig):
    entered, finish = threading.Event(), threading.Event()
    original = rig.vision.classify
    def blocked(frames):
        entered.set()
        assert finish.wait(3)
        return original(frames)
    rig.vision.classify = blocked
    tick_result = []
    th = threading.Thread(target=lambda: tick_result.append(tick(rig)))
    th.start(); assert entered.wait(2)
    assert "ignored" in tick(rig)
    rig.c.control("stop")
    finish.set(); th.join(3)
    assert not th.is_alive() and not rig.robot.jobs
    assert "ignored" in tick_result[0]
    assert rig.c.enabled is False


def test_disabled_and_rate_limited_calls_do_not_use_claude(rig):
    tick(rig)
    assert "ignored" in rig.c.analyze([PNG], rig.c.generation)
    assert rig.vision.calls == 1
    rig.c.configure(False, False)
    assert "ignored" in tick(rig)
    assert rig.vision.calls == 1


@pytest.mark.parametrize("frames", [None, [], [PNG]*4, ["https://example.com/image"], ["data:image/jpeg;base64,AAAA"], ["data:image/png;base64,?"], [3]])
def test_bad_image_inputs_are_rejected(frames):
    with pytest.raises(ValueError):
        image_blocks(frames)


def test_claude_receives_images_and_returns_only_the_gesture_fields():
    create = Mock(return_value=SimpleNamespace(content=[SimpleNamespace(type="tool_use", name="classify_gesture",
                    input={"gesture":"thumb_right_angle", "confidence":.95, "left_hand_on_brown_table":True,
                           "observation":"Five left fingers on brown table, thumb spread wide.", "instrument":"malicious"})]))
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    answer = ClaudeVision(client=client).classify([PNG,PNG])
    assert answer["gesture"] == "thumb_right_angle" and "instrument" not in answer
    kwargs = create.call_args.kwargs
    assert len([b for b in kwargs["messages"][0]["content"] if b["type"] == "image"]) == 2
    assert kwargs["tool_choice"]["type"] == "auto"  # this model rejects forced tool_choice


@pytest.mark.parametrize("score", [True, float("nan"), float("inf"), -1, 2, "1"])
def test_bad_model_confidence_is_rejected(score):
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw:SimpleNamespace(content=[SimpleNamespace(
        type="tool_use", name="classify_gesture", input={"gesture":"five_fingers", "confidence":score,"left_hand_on_brown_table":True})])))
    with pytest.raises(VisionUnavailable):
        ClaudeVision(client=client).classify([PNG])


def test_vision_rejects_a_pose_without_the_left_hand_on_the_brown_table():
    client=SimpleNamespace(messages=SimpleNamespace(create=lambda **kw:SimpleNamespace(content=[SimpleNamespace(
        type="tool_use",name="classify_gesture",input={"gesture":"five_fingers","confidence":.99,
        "left_hand_on_brown_table":False,"observation":"Only the right thumb is visible."})])))
    answer=ClaudeVision(client=client).classify([PNG])
    assert answer["gesture"]=="none" and answer["left_hand_on_brown_table"] is False


def test_simulated_worker_busy_release_stop_and_reset():
    r = Robot(Camera(None))
    try:
        with pytest.raises(ValueError):
            r.submit("scissors_potts")
        assert r.submit("scalpel_10")
        assert not r.submit("scalpel_20")
        deadline=time.monotonic()+3
        while r.snapshot()["phase"] != "presenting" and time.monotonic()<deadline:
            time.sleep(.02)
        r.control("release")
        r._thread.join(2)
        assert not r.snapshot()["busy"] and r.snapshot()["tool"] is None
        assert r.submit("scalpel_20")
        r.control("stop"); r._thread.join(2)
        assert r.snapshot()["stopped"] and r.snapshot()["phase"] == "stopped"
        assert not r.submit("scalpel_10")
        r.control("reset")
        assert r.snapshot()["phase"] == "idle"
    finally:
        r.close()


def test_real_adapter_calls_existing_camera_checked_picker(monkeypatch):
    from gesture import picker as pick
    import scrub_arm.app
    camera = Camera(99)
    camera.frame = lambda: object()
    app = SimpleNamespace(driver=SimpleNamespace(start=Mock(),enable=Mock(),clear_abort=Mock(),fault=None),
                          catalog=SimpleNamespace(resolve=lambda tool:tool))
    monkeypatch.setattr(scrub_arm.app,"build",lambda:app)
    monkeypatch.setattr(scrub_arm.app,"check_poses",lambda app:[])
    monkeypatch.setattr(pick,"setup",lambda **kw:None)
    monkeypatch.setattr(pick,"app",app)
    monkeypatch.setattr(pick,"d",SimpleNamespace(fault=None,clear_abort=Mock(),move_pose=Mock(),stop=Mock(),sleep=Mock()))
    monkeypatch.setattr(pick,"a",SimpleNamespace(_present=Mock(),_let_go=Mock(),stop=Mock()))
    pick_call, take_call = Mock(return_value=True), Mock(return_value="taken")
    monkeypatch.setattr(pick,"pick",pick_call)
    monkeypatch.setattr(pick,"wait_until_taken",take_call)
    r=Robot(camera,real=True)
    assert r.submit("forceps_fine_vascular")
    r._thread.join(2)
    pick_call.assert_called_once_with("forceps_fine_vascular")
    pick.a._present.assert_called_once_with("forceps_fine_vascular")
    pick.a._let_go.assert_called_once()
    assert take_call.call_args.kwargs["release_event"] is r._release
    assert r.snapshot()["phase"]=="idle"


def test_stop_during_physical_take_wait_never_opens_the_jaws(monkeypatch):
    from gesture import picker as pick
    import scrub_arm.app
    entered=threading.Event()
    camera=Camera(99);camera.frame=lambda:object()
    driver=SimpleNamespace(start=Mock(),enable=Mock(),clear_abort=Mock(),fault=None)
    app=SimpleNamespace(driver=driver,catalog=SimpleNamespace(resolve=lambda tool:tool))
    monkeypatch.setattr(scrub_arm.app,"build",lambda:app)
    monkeypatch.setattr(scrub_arm.app,"check_poses",lambda app:[])
    monkeypatch.setattr(pick,"setup",lambda **kw:None)
    monkeypatch.setattr(pick,"app",app)
    monkeypatch.setattr(pick,"d",SimpleNamespace(fault=None,clear_abort=Mock(),move_pose=Mock(),sleep=Mock()))
    monkeypatch.setattr(pick,"a",SimpleNamespace(_present=Mock(),_let_go=Mock(),stop=Mock()))
    monkeypatch.setattr(pick,"pick",lambda resolved:True)
    def wait(*args,release_event,**kwargs):
        entered.set();assert release_event.wait(2);return "enter"
    monkeypatch.setattr(pick,"wait_until_taken",wait)
    r=Robot(camera,real=True)
    assert r.submit("scalpel_10") and entered.wait(2)
    r.control("stop");r._thread.join(2)
    pick.a._let_go.assert_not_called()
    assert r.snapshot()["stopped"] and r.snapshot()["phase"]=="stopped"


def test_http_origin_and_payload_guards_and_key_stays_on_server(rig):
    server=serve(rig.c,port=0)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    base=f"http://127.0.0.1:{server.server_address[1]}"
    def request(path,body=None,headers=None):
        return urllib.request.urlopen(urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),
                                      headers={"Content-Type":"application/json",**(headers or {})}),timeout=3)
    try:
        state=json.load(request("/api/state"))
        assert "api_key" not in state
        html=request("/").read()
        assert b"Doctor view" in html and b"Robot view" in html
        assert b"DemoCameraSource" in request("/camera.js").read()
        for headers in [{"Origin":"https://other-site.example"},{"Host":"evil.example"}]:
            with pytest.raises(urllib.error.HTTPError) as e:
                request("/api/configure",{"enabled":True},headers)
            assert e.value.code==403
        with pytest.raises(urllib.error.HTTPError) as e:
            request("/api/analyze",{"frames":["bad"],"generation":rig.c.generation})
        assert e.value.code==400 and rig.vision.calls==0
        assert json.load(request("/api/configure",{"enabled":False,"auto_pick":False}))["enabled"] is False
    finally:
        server.shutdown();server.server_close()
