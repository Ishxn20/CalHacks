"""Episode recording for world-model training data.

One episode = one arm job. Per frame we keep the de-identified wrist image and the arm's state and
action at that instant, the (observation, action) pairs an action-conditioned video world model
trains on:

    video.mp4      wrist camera, faces blurred on-device, `fps` frames per second
    frames.jsonl   one line per video frame:
                   {"i", "t", "phase", "qDeg"[6], "qCmdDeg"[6], "gripperDeg", "gripperCmdDeg", "faces"}
                   observation = qDeg + gripperDeg, action = qCmdDeg + gripperCmdDeg
    keyframes      ~8 JPEGs (evenly spaced + every phase change) for the Claude labeller
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

import logging

from .deid import DeidUnavailable, FaceBlur

log = logging.getLogger(__name__)

FrameSource = Callable[[], "np.ndarray | None"]


def open_camera(index: int, width: int = 960, height: int = 540) -> FrameSource:
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    lock = threading.Lock()

    def read() -> np.ndarray | None:
        with lock:
            ok, frame = cap.read()
        return frame if ok else None

    return read


def synthetic_camera(width: int = 640, height: int = 360) -> FrameSource:
    """For simulated arms with no camera: a placeholder frame that says so (Claude will flag it)."""
    def read() -> np.ndarray:
        img = np.full((height, width, 3), 40, np.uint8)
        cv2.putText(img, "SIMULATED ARM - NO CAMERA", (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (200, 200, 200), 2)
        cv2.putText(img, time.strftime("%H:%M:%S"), (20, height // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 2)
        return img

    return read


@dataclass
class Episode:
    dir: Path
    fps: float
    frames: list[dict] = field(default_factory=list)
    keyframe_idx: list[int] = field(default_factory=list)
    faces: int = 0
    writer: cv2.VideoWriter | None = None


class Recorder:
    """Grabs frames at `fps` while an episode is open, pairing each with the latest arm state."""

    def __init__(self, camera: FrameSource, fps: float = 10.0, max_keyframes: int = 8, blur=None) -> None:
        self.camera, self.fps, self.max_keyframes = camera, fps, max_keyframes
        self._blur = blur
        self._state: dict | None = None
        self._ep: Episode | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def update_state(self, arm_status: dict) -> None:
        with self._lock:
            self._state = arm_status

    @property
    def recording(self) -> bool:
        return self._ep is not None

    def start(self) -> bool:
        """Open an episode. Refuses (returns False) if faces can't be blurred: no video without de-identification."""
        if self._blur is None:
            try:
                self._blur = FaceBlur()
            except (DeidUnavailable, cv2.error) as e:
                log.error("not recording: %s", e)
                return False
        self._ep = Episode(dir=Path(tempfile.mkdtemp(prefix="episode_")), fps=self.fps)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        return True

    def _run(self) -> None:
        ep = self._ep
        assert ep is not None
        period, t0 = 1.0 / self.fps, time.time()
        last_phase = None
        while not self._stop.is_set():
            frame = self.camera()
            if frame is not None:
                with self._lock:
                    st = self._state or {}
                arm = st.get("arm", {})
                faces = self._blur(frame)
                if ep.writer is None:
                    h, w = frame.shape[:2]
                    ep.writer = cv2.VideoWriter(str(ep.dir / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                ep.writer.write(frame)
                i = len(ep.frames)
                phase = st.get("phase", "unknown")
                ep.frames.append({
                    "i": i, "t": round(time.time(), 3), "phase": phase,
                    "qDeg": arm.get("q_deg", []), "qCmdDeg": arm.get("q_cmd_deg", []),
                    "gripperDeg": arm.get("gripper_deg"), "gripperCmdDeg": arm.get("gripper_cmd_deg"),
                    "faces": faces,
                })
                ep.faces += faces
                if phase != last_phase:                  # always keep the first frame of each phase
                    ep.keyframe_idx.append(i)
                    cv2.imwrite(str(ep.dir / f"kf_{i:05d}.jpg"), _small(frame), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    last_phase = phase
                elif i % max(1, int(self.fps * 2)) == 0:   # and a candidate every 2 s
                    cv2.imwrite(str(ep.dir / f"kf_{i:05d}.jpg"), _small(frame), [cv2.IMWRITE_JPEG_QUALITY, 85])
            t0 += period
            time.sleep(max(0.0, t0 - time.time()))

    def stop(self) -> Episode | None:
        """Close the episode. Returns it with video.mp4 / frames.jsonl written, or None if nothing was open."""
        ep = self._ep
        if ep is None:
            return None
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if ep.writer is not None:
            ep.writer.release()
        (ep.dir / "frames.jsonl").write_text("".join(json.dumps(f) + "\n" for f in ep.frames))
        ep.keyframe_idx = choose_keyframes(len(ep.frames), ep.keyframe_idx,
                                           [int(p.stem[3:]) for p in ep.dir.glob("kf_*.jpg")], self.max_keyframes)
        self._ep = None
        return ep


def choose_keyframes(n_frames: int, phase_starts: list[int], available: list[int], k: int) -> list[int]:
    """Phase changes first (they carry the story), then fill evenly from the other saved candidates."""
    if n_frames == 0 or not available:
        return []
    avail = sorted(set(available))
    chosen = [i for i in phase_starts if i in avail][:k]
    rest = [i for i in avail if i not in chosen]
    if len(chosen) < k and rest:
        picks = np.linspace(0, len(rest) - 1, min(k - len(chosen), len(rest))).round().astype(int)
        chosen += [rest[p] for p in picks]
    return sorted(set(chosen))


def _small(frame: np.ndarray, max_side: int = 768) -> np.ndarray:
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else frame
