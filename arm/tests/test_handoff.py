import math
import threading
import time

import numpy as np
import pytest

from scrub_arm.handoff import Busy, Phase
from scrub_arm.verify import VerifyResult


def wait_phase(asst, phase, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if asst.phase == phase:
            return
        time.sleep(0.005)
    raise AssertionError(f"never reached {phase}; at {asst.phase} {asst.status()['last_job']}")


def j6_deg(app):
    return math.degrees(app.driver.q_command()[-1])


def test_full_handoff_backhand(app):
    asst = app.assistant
    r = app.catalog.resolve("needle holder", "prolene", "6-0", "backhand")
    asst.submit("handoff", r)
    wait_phase(asst, Phase.PRESENTING)
    assert asst.holding.slot.id == "A1"
    # handoff pose wrist is -90; backhand rotates it 180 -> +90
    assert j6_deg(app) == pytest.approx(90.0, abs=0.5)
    asst.release()
    job = asst.wait_idle()
    assert job["status"] == "done", job
    assert asst.holding is None and asst.phase == Phase.IDLE
    assert np.allclose(app.driver.q_command(), app.poses.get("home").q)


def test_forehand_keeps_taught_wrist(app):
    asst = app.assistant
    asst.submit("handoff", app.catalog.resolve("castroviejo", orientation="forehand"))
    wait_phase(asst, Phase.PRESENTING)
    assert j6_deg(app) == pytest.approx(-90.0, abs=0.5)
    asst.release()
    asst.wait_idle()


def test_release_said_during_swing_is_latched(app):
    asst = app.assistant
    app.cfg.motion.speed_scale = 2.0
    asst.submit("handoff", app.catalog.resolve("debakey"))
    wait_phase(asst, Phase.MOVING)
    assert asst.release()["status"] in ("will release on arrival", "releasing")
    job = asst.wait_idle()
    assert job["status"] == "done" and "release command" in job["result"]


def test_missed_grasp_reports_and_does_not_present(app):
    app.driver.bus.object_in_jaws = False
    asst = app.assistant
    asst.submit("handoff", app.catalog.resolve("potts"))
    job = asst.wait_idle()
    assert job["status"] == "failed" and "missed the grasp" in job["error"]
    assert asst.holding is None


def test_stage_then_handoff_same_instrument_skips_pick(app):
    asst = app.assistant
    r = app.catalog.resolve("side biter")
    asst.submit("stage", r)
    assert asst.wait_idle()["status"] == "done"
    assert asst.phase == Phase.STAGED
    asst.submit("handoff", r)
    wait_phase(asst, Phase.PRESENTING)
    assert any("already holding" in e["msg"] for e in asst.events)
    asst.release()
    assert asst.wait_idle()["status"] == "done"


def test_stage_wrong_prediction_puts_it_back_first(app):
    asst = app.assistant
    asst.submit("stage", app.catalog.resolve("cross clamp"))
    asst.wait_idle()
    asst.submit("handoff", app.catalog.resolve("probe"))
    wait_phase(asst, Phase.PRESENTING)
    assert asst.holding.slot.id == "B4"
    assert any("putting back" in e["msg"] for e in asst.events)
    asst.release()
    asst.wait_idle()


def test_verification_failure_returns_instrument(app):
    asst = app.assistant

    class WrongTool:
        enabled = True

        def verify(self, expected, slot):
            return VerifyResult(ok=False, detected="scissors_potts", confidence=0.9, note="tray-cam saw scissors_potts")

    asst.verifier = WrongTool()
    asst.submit("handoff", app.catalog.resolve("debakey"))
    job = asst.wait_idle()
    assert job["status"] == "failed" and "scissors_potts" in job["error"]
    assert asst.holding is None
    assert any("tray-cam" in e["msg"] for e in asst.events)


def test_tug_releases(app):
    app.cfg.handoff.tug.enabled = True
    asst = app.assistant
    asst.submit("handoff", app.catalog.resolve("potts"))
    wait_phase(asst, Phase.PRESENTING)
    time.sleep(0.5)  # let the baseline settle
    app.driver.bus.external_offset[5] = math.radians(8)  # surgeon pulls; wrist deflects
    job = asst.wait_idle()
    app.driver.bus.external_offset[:] = 0
    assert job["status"] == "done" and "tug" in job["result"]


def test_stop_freezes_then_reset_recovers(app):
    asst = app.assistant
    app.cfg.motion.speed_scale = 1.0
    asst.submit("handoff", app.catalog.resolve("potts"))
    time.sleep(0.15)
    asst.stop()
    job = asst.wait_idle()
    assert job["status"] == "stopped"
    q_frozen = app.driver.q_command()
    time.sleep(0.1)
    assert np.allclose(app.driver.q_command(), q_frozen)
    asst.reset()
    app.cfg.motion.speed_scale = 20.0
    asst.submit("home")
    assert asst.wait_idle()["status"] == "done"


def test_one_job_at_a_time(app):
    asst = app.assistant
    asst.submit("handoff", app.catalog.resolve("potts"))
    with pytest.raises(Busy):
        asst.submit("home")
    wait_phase(asst, Phase.PRESENTING)
    asst.release()
    asst.wait_idle()


def test_present_timeout_keeps_holding_then_release(app):
    app.cfg.handoff.present_timeout_s = 0.2
    asst = app.assistant
    asst.submit("handoff", app.catalog.resolve("potts"))
    job = asst.wait_idle()
    assert "still holding" in job["result"]
    assert asst.holding is not None
    asst.release()
    assert asst.wait_idle()["status"] == "done"
    assert asst.holding is None
