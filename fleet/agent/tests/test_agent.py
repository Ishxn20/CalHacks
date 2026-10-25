import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from scrub_fleet.agent import Agent
from scrub_fleet.armcfg import ConfigChangeError, apply_change, current_value, excerpt
from scrub_fleet.export import export, frames_to_arrays
from scrub_fleet.recorder import Recorder, choose_keyframes
from scrub_fleet.speak import confirm_line

ARM_YAML = """\
channel: can0
gripper:
  move_torque_nm: 1.5    # cap while opening
  grip_torque_nm: 0.8
motion:
  speed_scale: 0.5             # global multiplier
  approach_speed_scale: 0.25
  tracking_error_limit_deg: 25.0
pick:
  contact_deviation_deg: 1.5   # joint lag
server:
  port: 8765
"""


@pytest.fixture
def arm_yaml(tmp_path) -> Path:
    p = tmp_path / "arm.yaml"
    p.write_text(ARM_YAML)
    return p


# ── arm.yaml changes ──────────────────────────────────────────────────────

def test_apply_change_edits_one_value_and_keeps_comments(arm_yaml):
    msg = apply_change(arm_yaml, "motion.speed_scale", 0.35)
    text = arm_yaml.read_text()
    assert "speed_scale: 0.35             # global multiplier" in text
    assert "approach_speed_scale: 0.25" in text           # the similarly named key is untouched
    assert text.replace("0.35", "0.5", 1) == ARM_YAML
    assert "restart" in msg and list(arm_yaml.parent.glob("arm.yaml.bak-*"))


@pytest.mark.parametrize("key,to", [("motion.speed_scale", 0.9), ("motion.speed_scale", -1),
                                    ("channel", 1), ("joints.limits", 170)])
def test_apply_change_refuses_out_of_bounds_or_unlisted(arm_yaml, key, to):
    with pytest.raises(ConfigChangeError):
        apply_change(arm_yaml, key, to)
    assert arm_yaml.read_text() == ARM_YAML


def test_key_in_a_different_section_is_not_matched(arm_yaml):
    # gripper has no speed_scale; must not grab motion's.
    with pytest.raises(ConfigChangeError):
        current_value(arm_yaml, "gripper.speed_scale")


def test_missing_setting_is_reported(arm_yaml):
    with pytest.raises(ConfigChangeError, match="predate"):
        apply_change(arm_yaml, "gripper.holding_min_gap_deg", 2.0)


def test_excerpt(arm_yaml):
    ex = excerpt(arm_yaml)
    assert ex["motion.speed_scale"] == 0.5 and ex["gripper.holding_min_gap_deg"] is None


# ── recording ─────────────────────────────────────────────────────────────

def test_keyframes_prefer_phase_changes_then_fill_evenly():
    k = choose_keyframes(100, [0, 40, 90], [0, 20, 40, 60, 80, 90, 99], 5)
    assert {0, 40, 90} <= set(k) and len(k) == 5


def test_keyframes_empty():
    assert choose_keyframes(0, [], [], 8) == []


def test_recorder_writes_video_frames_and_keyframes():
    frame = np.full((120, 160, 3), 90, np.uint8)
    rec = Recorder(lambda: frame.copy(), fps=20, blur=lambda f: 0)
    rec.update_state({"phase": "picking", "arm": {"q_deg": [1, 2, 3, 4, 5, 6], "q_cmd_deg": [1, 2, 3, 4, 5, 7],
                                                  "gripper_deg": 100.0, "gripper_cmd_deg": 0.0}})
    rec.start()
    time.sleep(0.3)
    rec.update_state({"phase": "presenting", "arm": {}})
    time.sleep(0.3)
    ep = rec.stop()
    assert ep and len(ep.frames) >= 6
    assert (ep.dir / "video.mp4").stat().st_size > 0
    lines = (ep.dir / "frames.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["qCmdDeg"][5] == 7
    phases = [ep.frames[i]["phase"] for i in ep.keyframe_idx]
    assert "picking" in phases and "presenting" in phases
    assert all((ep.dir / f"kf_{i:05d}.jpg").exists() for i in ep.keyframe_idx)


# ── export ────────────────────────────────────────────────────────────────

def test_frames_to_arrays_aligns_observation_and_action():
    a = frames_to_arrays([
        {"t": 10.0, "phase": "a", "qDeg": [1, 2, 3, 4, 5, 6], "qCmdDeg": [1, 2, 3, 4, 5, 6.5], "gripperDeg": 50, "gripperCmdDeg": 0},
        {"t": 10.1, "phase": "b", "qDeg": [None] * 6, "qCmdDeg": [], "gripperDeg": None, "gripperCmdDeg": None},
    ])
    assert a["observation_state"].shape == (2, 7) and a["action"].shape == (2, 7)
    assert a["action"][0, 5] == 6.5 and np.isnan(a["observation_state"][1]).all()
    assert a["timestamp"][1] == pytest.approx(0.1)


def test_export_writes_lerobot_style_layout(tmp_path):
    frames = "\n".join(json.dumps({"t": 1 + i / 10, "phase": "p", "qDeg": [0] * 6, "qCmdDeg": [1] * 6,
                                   "gripperDeg": 0, "gripperCmdDeg": 0}) for i in range(5))
    blobs = {"f": frames.encode(), "v": b"mp4"}
    labels = {"task": "hand the scalpel", "caption": "c", "success": True, "failureMode": None,
              "segments": [], "model": "claude-opus-5-5"}
    n = export([{"id": "e1", "robotId": "r1", "request": "scalpel", "startedAt": 5, "fps": 10, "frameCount": 5,
                 "labels": labels, "framesUrl": "f", "videoUrl": "v"}], tmp_path, fetch=blobs.__getitem__)
    assert n == 1
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["total_frames"] == 5 and info["features"]["video"]["deidentified"]
    row = json.loads((tmp_path / "meta" / "episodes.jsonl").read_text())
    assert row["task"] == "hand the scalpel"
    assert np.load(tmp_path / "data" / "episode_000000.npz")["action"].shape == (5, 7)


# ── speech lines ──────────────────────────────────────────────────────────

def test_confirm_lines():
    assert confirm_line("handoff", {"say": "needle holder, backhand"}) == "needle holder, backhand, coming."
    assert confirm_line("handoff", {"error": "x", "say": "No needle holder on the tray"}) == "No needle holder on the tray"
    assert confirm_line("stop", {}) == "Stopped."


# ── agent loop against fakes ──────────────────────────────────────────────

class FakeArm:
    def __init__(self):
        self.st = {"phase": "idle", "job": None, "last_job": None, "events": [],
                   "arm": {"enabled": True, "fault": None, "q_deg": [0] * 6, "q_cmd_deg": [0] * 6, "gripper_deg": 0}}
        self.posts = []

    def status(self):
        return json.loads(json.dumps(self.st))

    def post(self, path, body=None):
        self.posts.append((path, body))
        return {"job": 1, "say": "scalpel"}


class FakeFleet:
    def __init__(self, commands=()):
        self.calls, self.queue, self.uploads = [], list(commands), 0

    def heartbeat(self, **b):
        self.calls.append(("heartbeat", b))
        return {"recording": True}

    def commands(self):
        q, self.queue = self.queue, []
        return q

    def command_result(self, cid, ok, result):
        self.calls.append(("result", cid, ok, result))

    def start_episode(self, *a):
        self.calls.append(("start", a))
        return {"episodeId": "ep1", "consent": True}

    def upload(self, data, ct):
        self.uploads += 1
        return f"s{self.uploads}"

    def finish_episode(self, **b):
        self.calls.append(("finish", b))

    def incident(self, fault, phase, context):
        self.calls.append(("incident", fault, phase, context))
        return "inc1"


class Mute:
    def __init__(self):
        self.lines = []

    def say(self, text, urgent=False):
        self.lines.append(text)


def make_agent(arm_yaml, commands=(), camera=True):
    arm, fleet, spk = FakeArm(), FakeFleet(commands), Mute()
    rec = Recorder(lambda: np.full((90, 120, 3), 80, np.uint8), fps=20, blur=lambda f: 0) if camera else None
    return Agent(fleet, arm, rec, spk, arm_yaml), arm, fleet, spk


def test_one_job_becomes_one_uploaded_episode(arm_yaml):
    agent, arm, fleet, _ = make_agent(arm_yaml)
    agent.tick()
    arm.st.update(phase="picking", job={"id": 7, "kind": "handoff", "request": "scalpel", "started": time.time()})
    agent.tick()
    time.sleep(0.4)
    arm.st.update(phase="idle", job=None, last_job={"id": 7, "status": "done", "error": None, "finished": time.time()})
    agent.tick()
    kinds = [c[0] for c in fleet.calls]
    assert kinds.count("start") == 1 and kinds.count("finish") == 1
    fin = next(c[1] for c in fleet.calls if c[0] == "finish")
    assert fin["outcome"] == "done" and fin["frameCount"] > 0 and fin["videoId"] and fin["keyframeIds"]


def test_fault_opens_one_incident_and_speaks(arm_yaml):
    agent, arm, fleet, spk = make_agent(arm_yaml, camera=False)
    arm.st["arm"]["fault"] = "joint3 lagged its setpoint by 31 deg (collision guard)"
    arm.st["phase"] = "fault"
    agent.tick()
    agent.tick()
    incidents = [c for c in fleet.calls if c[0] == "incident"]
    assert len(incidents) == 1
    assert incidents[0][3]["config"]["motion.speed_scale"] == 0.5
    assert any("stopped" in line for line in spk.lines)


def test_commands_go_to_the_arm_and_config_changes_to_arm_yaml(arm_yaml):
    agent, arm, fleet, spk = make_agent(arm_yaml, commands=[
        {"id": "c1", "kind": "handoff", "text": "scalpel", "change": None},
        {"id": "c2", "kind": "apply_config", "text": None, "change": {"key": "motion.speed_scale", "to": 0.3}},
        {"id": "c3", "kind": "apply_config", "text": None, "change": {"key": "motion.speed_scale", "to": 5}},
        {"id": "c4", "kind": "self_destruct", "text": None, "change": None},
    ], camera=False)
    agent.tick()
    assert arm.posts == [("/handoff", {"text": "scalpel"})]
    results = {c[1]: c[2] for c in fleet.calls if c[0] == "result"}
    assert results == {"c1": True, "c2": True, "c3": False, "c4": False}
    assert current_value(arm_yaml, "motion.speed_scale") == 0.3
    assert "scalpel, coming." in spk.lines


def test_arm_down_is_reported_not_fatal(arm_yaml):
    agent, arm, fleet, _ = make_agent(arm_yaml, camera=False)
    arm.status = lambda: (_ for _ in ()).throw(OSError("connection refused"))
    agent.tick()
    assert any(c[0] == "incident" and "unreachable" in c[1] for c in fleet.calls)


def test_face_detector_loads_and_finds_nothing_on_a_blank_frame():
    from scrub_fleet.deid import FaceBlur

    img = np.zeros((240, 320, 3), np.uint8)
    assert FaceBlur()(img) == 0


def test_no_detector_means_no_recording(monkeypatch, tmp_path):
    import scrub_fleet.deid as deid

    monkeypatch.setattr(deid, "CACHE", tmp_path)
    monkeypatch.setattr(deid, "MODEL_URL", "http://127.0.0.1:9/nope.onnx")
    rec = Recorder(lambda: np.zeros((10, 10, 3), np.uint8))
    assert rec.start() is False and not rec.recording


def test_tampered_model_is_rejected(monkeypatch, tmp_path):
    import scrub_fleet.deid as deid

    monkeypatch.setattr(deid, "CACHE", tmp_path)
    (tmp_path / "face_detection_yunet_2023mar.onnx").write_bytes(b"not the model")
    with pytest.raises(deid.DeidUnavailable, match="checksum"):
        deid.model_path()
