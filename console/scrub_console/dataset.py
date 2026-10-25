"""Every arm job becomes a training episode for world models of surgical manipulation.

The bridge hands each arm /status to `DatasetRecorder.observe`. When a handoff / stage / return starts, the
recorder captures the wrist camera with faces blurred on-device (no blur model, no recording); when the job
ends, the clip, its per-frame state/action table and keyframes are uploaded to Convex file storage from a
background thread, and Convex has Claude label it (convex/labeling.ts). Nothing here can slow the arm down.
Recording is opt-in per case (`--record`): turning it on is the consent for research use.
"""
from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import urllib.request
from typing import Callable

from .recorder import Episode, Recorder

log = logging.getLogger(__name__)
RECORDED_JOBS = ("handoff", "stage", "return")


class DatasetRecorder:
    def __init__(self, recorder: Recorder, convex, emit: Callable[..., None], simulated: bool = False) -> None:
        self.recorder, self.convex, self.emit, self.simulated = recorder, convex, emit, simulated
        self.job: dict | None = None             # the arm job being recorded
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._uploader, name="episode-upload", daemon=True).start()

    def observe(self, status: dict) -> None:
        self.recorder.update_state(status)
        job = status.get("job")
        if self.job is None and job and job.get("kind") in RECORDED_JOBS:
            if self.recorder.start():
                self.job = job
            else:
                self.emit("episode", job=job.get("id"), status="not_recorded", reason="faces can't be blurred, so nothing is recorded")
        elif self.job is not None and (job is None or job.get("id") != self.job["id"]):
            last = status.get("last_job") or {}
            ep = self.recorder.stop()
            if ep is not None:
                self._q.put((self.job, last if last.get("id") == self.job["id"] else {}, ep))
            self.job = None

    def _uploader(self) -> None:
        while True:
            job, last, ep = self._q.get()
            try:
                self.emit("episode", job=job["id"], status="uploaded", frames=len(ep.frames), faces=ep.faces,
                          id=self.upload(job, last, ep))
            except Exception as e:  # noqa: BLE001 - a dataset problem must never reach the arm
                log.warning("episode upload failed; keeping %s: %s", ep.dir, e)
                self.emit("episode", job=job["id"], status="upload_failed", reason=str(e))
            else:
                shutil.rmtree(ep.dir, ignore_errors=True)

    def _store(self, data: bytes, ctype: str) -> str:
        url = self.convex.mutation("episodes:uploadUrl", {})
        req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": ctype})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["storageId"]

    def upload(self, job: dict, last: dict, ep: Episode) -> str:
        video = ep.dir / "video.mp4"
        has_video = bool(ep.frames) and video.exists()
        t0 = ep.frames[0]["t"] if ep.frames else 0.0
        return self.convex.mutation("episodes:record", {
            "jobId": job["id"], "kind": job["kind"], "request": job.get("request"),
            "startedAt": (job.get("started") or t0) * 1000, "endedAt": (last.get("finished") or time.time()) * 1000,
            "outcome": last.get("status", "unknown"), "error": last.get("error"),
            "videoId": self._store(video.read_bytes(), "video/mp4") if has_video else None,
            "framesId": self._store((ep.dir / "frames.jsonl").read_bytes(), "application/x-ndjson") if has_video else None,
            "keyframeIds": [self._store((ep.dir / f"kf_{i:05d}.jpg").read_bytes(), "image/jpeg") for i in ep.keyframe_idx]
            if has_video else [],
            "keyframeTimesS": [round(ep.frames[i]["t"] - t0, 2) for i in ep.keyframe_idx] if has_video else [],
            "frameCount": len(ep.frames), "fps": ep.fps, "facesBlurred": ep.faces,
            "consent": True, "simulated": self.simulated,
        })
