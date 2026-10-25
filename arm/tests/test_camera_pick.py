"""The shared wrist-camera pick and hand-off, with fake camera observations and simulated motors."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scrub_arm.camera_pick import GONE_MIN_PX, CameraPickup
from scrub_arm.driver import MotionAborted
from scrub_arm.handoff import Phase, TaskFailed


def detection(x=450):
    return SimpleNamespace(fine_id="scalpel_10", alt_ids=[], candidates=[], exposure=1, confidence=1,
                           grasp_mm=(x, 0), yaw_deg=0, thickness_mm=4, width_mm=13)


def camera(app):
    app.catalog.require_slot = False
    c = CameraPickup.__new__(CameraPickup)      # no real camera or detector
    c.a, c.d = app.assistant, app.driver
    c.d.gripper_open = Mock()
    c.a._pick_at = Mock()
    return c


def test_confirmed_miss_rescans_and_a_good_pick_keeps_the_jaws_closed(app):
    c = camera(app)
    r = app.catalog.resolve("10 blade")
    c.look = Mock(side_effect=[[detection()], [detection()], [detection()], []])   # miss, then held
    assert c.pick(r)
    assert c.a._pick_at.call_count == 2 and c.a.holding == r
    assert c.d.gripper_open.call_count == 3      # start, after the miss, retry; never after success


def test_nothing_seen_gives_up_without_moving_to_pick(app):
    c = camera(app)
    c.look = Mock(return_value=[])
    assert not c.pick(app.catalog.resolve("10 blade"))
    c.a._pick_at.assert_not_called()


def test_camera_failure_after_lift_keeps_jaws_closed(app):
    c = camera(app)
    c.look = Mock(side_effect=[[detection()], TaskFailed("the wrist camera gave no picture")])
    assert c.pick(app.catalog.resolve("10 blade"))
    assert c.d.gripper_open.call_count == 1


def test_handoff_wait_ends_on_release_command_and_on_operator_stop(app):
    c = camera(app)
    c.held_red = Mock(return_value=5000)
    c.a._release.set()
    assert c.wait_until_taken(timeout_s=1) == "release command"
    c.a._release.clear()
    app.driver.stop()
    with pytest.raises(MotionAborted):
        c.wait_until_taken(timeout_s=1)


def test_handoff_wait_sees_the_tool_taken(app):
    c = camera(app)
    base = int(GONE_MIN_PX * 2)
    c.held_red = Mock(side_effect=[base] * 5 + [0] * 1000)
    assert c.wait_until_taken(timeout_s=5) == "taken"


def test_http_handoff_uses_camera_pick_and_puts_it_back_on_timeout(app):
    a = app.assistant
    app.catalog.require_slot = False
    a.cfg.pick.mode = "ik"
    r = app.catalog.resolve("10 blade")

    def pick(resolved):
        a.holding = resolved
        a._picked_from = (450, 0, 0, 4)
        return True
    a.camera_pick = SimpleNamespace(pick=Mock(side_effect=pick), wait_until_taken=Mock(return_value="timeout"))
    a._present = Mock()
    a._place_at = Mock(side_effect=lambda *args, **kw: setattr(a, "holding", None))
    a.submit("handoff", r)
    job = a.wait_idle()
    assert job["status"] == "done" and job["result"].startswith("returned"), job
    a.camera_pick.pick.assert_called_once_with(r)
    a._place_at.assert_called_once_with(450, 0, 0, 4, sense_contact=True)
    assert a.phase == Phase.IDLE and a.holding is None


def test_http_handoff_fails_cleanly_when_the_camera_pick_gives_up(app):
    a = app.assistant
    app.catalog.require_slot = False
    a.cfg.pick.mode = "ik"
    a.camera_pick = SimpleNamespace(pick=Mock(return_value=False))
    a.submit("handoff", app.catalog.resolve("10 blade"))
    job = a.wait_idle()
    assert job["status"] == "failed" and "couldn't pick up" in job["error"]
    assert a.phase == Phase.IDLE
