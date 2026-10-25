import json
import time

import numpy as np
import pytest

from scrub_console.dataset import DatasetRecorder
from scrub_console.export import export, frames_to_arrays
from scrub_console.recorder import Recorder, choose_keyframes


def status(phase, job=None, last=None, q_cmd=6):
    return {"phase": phase, "job": job, "last_job": last,
            "arm": {"q_deg": [1, 2, 3, 4, 5, 6], "q_cmd_deg": [1, 2, 3, 4, 5, q_cmd], "gripper_deg": 100.0, "gripper_cmd_deg": 0.0}}


class FakeConvex:
    def __init__(self):
        self.calls = []

    def mutation(self, name, args):
        self.calls.append((name, args))
        return "episode-1"


# ── one arm job becomes one uploaded episode ────────────────────────────

def test_an_arm_job_is_recorded_blurred_and_uploaded_with_its_labels_queued():
    convex, events = FakeConvex(), []
    rec = Recorder(lambda: np.full((120, 160, 3), 90, np.uint8), fps=20, blur=lambda f: 0)
    ds = DatasetRecorder(rec, convex, lambda kind, /, **d: events.append((kind, d)), simulated=True)
    stored = []
    ds._store = lambda data, ctype: stored.append(ctype) or f"s{len(stored)}"
    job = {"id": 7, "kind": "handoff", "request": "castroviejo, backhand", "started": time.time()}
    ds.observe(status("idle"))
    ds.observe(status("picking", job=job))
    assert rec.recording
    time.sleep(0.3)
    ds.observe(status("presenting", job=job))
    time.sleep(0.3)
    ds.observe(status("idle", last={**job, "status": "done", "finished": time.time()}))
    deadline = time.time() + 5
    while not any(k == "episode" for k, _ in events) and time.time() < deadline:
        time.sleep(0.05)
    name, ep = convex.calls[-1]
    assert name == "episodes:record"
    assert ep["jobId"] == 7 and ep["outcome"] == "done" and ep["consent"] is True and ep["simulated"] is True
    assert ep["videoId"] and ep["framesId"] and len(ep["keyframeIds"]) >= 2 and ep["frameCount"] >= 6
    assert stored[:2] == ["video/mp4", "application/x-ndjson"]
    assert events[-1][1]["status"] == "uploaded" and not rec.recording


def test_no_face_blur_means_the_job_is_not_recorded():
    events = []
    rec = Recorder(lambda: np.zeros((10, 10, 3), np.uint8))
    rec.start = lambda: False                         # what start() returns when the blur model can't load
    ds = DatasetRecorder(rec, FakeConvex(), lambda kind, /, **d: events.append((kind, d)))
    ds.observe(status("picking", job={"id": 1, "kind": "handoff"}))
    assert ds.job is None and events[0][1]["status"] == "not_recorded"


# ── recorder (from the fleet agent) ─────────────────────────────────────

def test_keyframes_prefer_phase_changes_then_fill_evenly():
    k = choose_keyframes(100, [0, 40, 90], [0, 20, 40, 60, 80, 90, 99], 5)
    assert {0, 40, 90} <= set(k) and len(k) == 5
    assert choose_keyframes(0, [], [], 8) == []


def test_recorder_writes_video_frames_and_keyframes():
    frame = np.full((120, 160, 3), 90, np.uint8)
    rec = Recorder(lambda: frame.copy(), fps=20, blur=lambda f: 0)
    rec.update_state(status("picking", q_cmd=7))
    rec.start()
    time.sleep(0.3)
    rec.update_state({"phase": "presenting", "arm": {}})
    time.sleep(0.3)
    ep = rec.stop()
    assert ep and len(ep.frames) >= 6
    assert (ep.dir / "video.mp4").stat().st_size > 0
    assert json.loads((ep.dir / "frames.jsonl").read_text().splitlines()[0])["qCmdDeg"][5] == 7
    phases = [ep.frames[i]["phase"] for i in ep.keyframe_idx]
    assert "picking" in phases and "presenting" in phases


def test_no_detector_means_no_recording(monkeypatch, tmp_path):
    import scrub_console.deid as deid

    monkeypatch.setattr(deid, "CACHE", tmp_path)
    monkeypatch.setattr(deid, "MODEL_URL", "http://127.0.0.1:9/nope.onnx")
    rec = Recorder(lambda: np.zeros((10, 10, 3), np.uint8))
    assert rec.start() is False and not rec.recording


def test_tampered_model_is_rejected(monkeypatch, tmp_path):
    import scrub_console.deid as deid

    monkeypatch.setattr(deid, "CACHE", tmp_path)
    (tmp_path / "face_detection_yunet_2023mar.onnx").write_bytes(b"not the model")
    with pytest.raises(deid.DeidUnavailable, match="checksum"):
        deid.model_path()


# ── export ──────────────────────────────────────────────────────────────

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
    n = export([{"id": "e1", "request": "scalpel", "startedAt": 5, "fps": 10, "frameCount": 5,
                 "labels": labels, "framesUrl": "f", "videoUrl": "v"}], tmp_path, fetch=blobs.__getitem__)
    assert n == 1
    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["total_frames"] == 5 and info["features"]["video"]["deidentified"]
    assert json.loads((tmp_path / "meta" / "episodes.jsonl").read_text())["task"] == "hand the scalpel"
    assert np.load(tmp_path / "data" / "episode_000000.npz")["action"].shape == (5, 7)
