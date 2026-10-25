"""Fleet agent: runs next to `scrub_arm serve` on each arm's computer.

    python -m scrub_fleet.agent --site-url $CONVEX_SITE_URL --token $ROBOT_TOKEN [--camera 0 | --no-camera]

Every tick it reads the arm's /status and:
  * streams downsampled telemetry + a heartbeat to the fleet,
  * records one episode per arm job (de-identified wrist video + state/action table) and uploads it,
  * runs commands an operator queued (the arm server still enforces its own safety),
  * applies approved config fixes (allowlisted, bounded, backed up),
  * opens an incident when the arm faults, so the Droid worker can diagnose it,
  * says out loud what it's doing (Cartesia).
"""
from __future__ import annotations

import argparse
import collections
import logging
import os
import shutil
import time
from pathlib import Path

from . import __version__
from .armcfg import ConfigChangeError, apply_change, excerpt
from .client import ArmClient, FleetClient, HttpError
from .recorder import Episode, Recorder, open_camera, synthetic_camera
from .speak import Speaker, confirm_line

log = logging.getLogger("scrub_fleet")
REPO = Path(__file__).resolve().parents[3]
ARM_COMMANDS = {"handoff", "stage", "release", "return", "home", "stop", "reset"}


def sample_from(status: dict) -> dict:
    arm = status.get("arm", {})
    return {"t": time.time() * 1000, "phase": status.get("phase", "unknown"),
            "qDeg": arm.get("q_deg", []), "qCmdDeg": arm.get("q_cmd_deg", []), "gripperDeg": arm.get("gripper_deg")}


class Agent:
    def __init__(self, fleet: FleetClient, arm: ArmClient, recorder: Recorder | None, speaker: Speaker,
                 arm_yaml: Path, tick_s: float = 0.5) -> None:
        self.fleet, self.arm, self.recorder, self.speaker = fleet, arm, recorder, speaker
        self.arm_yaml, self.tick_s = arm_yaml, tick_s
        self.samples: list[dict] = []
        self.history: collections.deque[dict] = collections.deque(maxlen=120)   # last ~60 s for incidents
        self.recording_allowed = True
        self.episode: dict | None = None          # {"id", "job_id", "started"} while a job runs
        self.reported_fault: str | None = None
        self.last_beat = 0.0

    # ── one tick ─────────────────────────────────────────────────────────
    def tick(self) -> None:
        try:
            status = self.arm.status()
        except (OSError, HttpError) as e:
            log.warning("arm server unreachable: %s", e)
            status = {"phase": "offline", "arm": {"fault": f"arm server unreachable: {e}"}, "job": None}
        if self.recorder:
            self.recorder.update_state(status)
        s = sample_from(status)
        self.samples.append(s)
        self.history.append({**s, "events": status.get("events", [])[-3:]})
        self._track_episode(status)
        self._watch_faults(status)
        if time.time() - self.last_beat >= 1.0:
            self._beat(status)
            self._run_commands()

    def _beat(self, status: dict) -> None:
        arm, job = status.get("arm", {}), status.get("job")
        try:
            reply = self.fleet.heartbeat(agentVersion=__version__, enabled=bool(arm.get("enabled")),
                                         fault=arm.get("fault"),
                                         job=f"{job['kind']} #{job['id']}" if job else None,
                                         samples=self.samples[::2][-50:])
            self.recording_allowed = bool(reply.get("recording", True))
            self.samples.clear()
            self.last_beat = time.time()
        except (OSError, HttpError) as e:
            log.warning("fleet unreachable (%s); buffering telemetry", e)
            self.samples = self.samples[-200:]

    # ── episodes ─────────────────────────────────────────────────────────
    def _track_episode(self, status: dict) -> None:
        job = status.get("job")
        if job and self.episode is None and job.get("kind") in ("handoff", "stage", "return"):
            self._start_episode(job)
        elif self.episode is not None and (job is None or job.get("id") != self.episode["job_id"]):
            last = status.get("last_job") or {}
            self._finish_episode(last if last.get("id") == self.episode["job_id"] else {})

    def _start_episode(self, job: dict) -> None:
        try:
            r = self.fleet.start_episode(job["id"], job["kind"], job.get("request"), job["started"] * 1000)
        except (OSError, HttpError) as e:
            log.warning("could not open episode: %s", e)
            return
        self.episode = {"id": r["episodeId"], "job_id": job["id"], "started": job["started"]}
        if self.recorder and r.get("consent") and self.recording_allowed and not self.recorder.start():
            self.speaker.say("Recording is off: I can't blur faces right now.")
        log.info("episode %s for %s #%s (%s)", r["episodeId"], job["kind"], job["id"],
                 "recording" if self.recorder and self.recorder.recording else "not recording")

    def _finish_episode(self, last_job: dict) -> None:
        ep_info, self.episode = self.episode, None
        assert ep_info is not None
        ep: Episode | None = self.recorder.stop() if self.recorder else None
        body = {"episodeId": ep_info["id"], "endedAt": (last_job.get("finished") or time.time()) * 1000,
                "outcome": last_job.get("status", "unknown"), "error": last_job.get("error"),
                "videoId": None, "framesId": None, "keyframeIds": [], "keyframeTimesS": [],
                "frameCount": 0, "fps": 0, "facesBlurred": 0}
        try:
            if ep and ep.frames and (ep.dir / "video.mp4").exists():
                t0 = ep.frames[0]["t"]
                body.update(
                    videoId=self.fleet.upload((ep.dir / "video.mp4").read_bytes(), "video/mp4"),
                    framesId=self.fleet.upload((ep.dir / "frames.jsonl").read_bytes(), "application/x-ndjson"),
                    keyframeIds=[self.fleet.upload((ep.dir / f"kf_{i:05d}.jpg").read_bytes(), "image/jpeg")
                                 for i in ep.keyframe_idx],
                    keyframeTimesS=[round(ep.frames[i]["t"] - t0, 2) for i in ep.keyframe_idx],
                    frameCount=len(ep.frames), fps=ep.fps, facesBlurred=ep.faces)
            self.fleet.finish_episode(**body)
            log.info("episode %s closed: %s, %d frames", ep_info["id"], body["outcome"], body["frameCount"])
        except (OSError, HttpError) as e:
            log.error("episode upload failed, keeping %s: %s", ep.dir if ep else "-", e)
            return
        if ep:
            shutil.rmtree(ep.dir, ignore_errors=True)
        if body["outcome"] in ("failed", "fault"):
            self.speaker.say(f"That didn't work: {body['error'] or body['outcome']}.", urgent=True)

    # ── faults → incidents ───────────────────────────────────────────────
    def _watch_faults(self, status: dict) -> None:
        fault = (status.get("arm") or {}).get("fault")
        if status.get("phase") == "fault" and not fault:
            fault = (status.get("last_job") or {}).get("error") or "arm reported a fault"
        if not fault:
            self.reported_fault = None
            return
        if fault == self.reported_fault:
            return
        self.speaker.say(f"I've stopped. {fault}", urgent=True)
        context = {"recent": list(self.history), "events": status.get("events", []),
                   "lastJob": status.get("last_job"), "config": excerpt(self.arm_yaml) if self.arm_yaml.exists() else {}}
        try:
            self.fleet.incident(fault, status.get("phase", "unknown"), context)
            self.reported_fault = fault
        except (OSError, HttpError) as e:
            log.warning("could not report incident: %s", e)

    # ── commands from the fleet ──────────────────────────────────────────
    def _run_commands(self) -> None:
        try:
            commands = self.fleet.commands()
        except (OSError, HttpError) as e:
            log.warning("could not fetch commands: %s", e)
            return
        for c in commands:
            ok, result = self._run_command(c)
            try:
                self.fleet.command_result(c["id"], ok, result)
            except (OSError, HttpError) as e:
                log.warning("could not report command result: %s", e)

    def _run_command(self, c: dict) -> tuple[bool, str]:
        kind = c["kind"]
        if kind == "apply_config":
            ch = c.get("change") or {}
            try:
                msg = apply_change(self.arm_yaml, ch["key"], float(ch["to"]))
            except (ConfigChangeError, KeyError, OSError) as e:
                return False, str(e)
            self.speaker.say("Configuration updated. Restart me to apply it.")
            return True, msg
        if kind not in ARM_COMMANDS:
            return False, f"unknown command {kind}"
        body = {"text": c["text"]} if kind in ("handoff", "stage") else {}
        reply = self.arm.post(f"/{kind}", body)
        self.speaker.say(confirm_line(kind, reply), urgent=kind == "stop")
        return "error" not in reply, str(reply.get("error") or reply)

    def run(self) -> None:
        log.info("fleet agent %s running; Ctrl-C to stop", __version__)
        while True:
            t = time.time()
            self.tick()
            time.sleep(max(0.0, self.tick_s - (time.time() - t)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="scrub_fleet.agent")
    ap.add_argument("--site-url", default=os.environ.get("CONVEX_SITE_URL"), help="Convex HTTP actions URL")
    ap.add_argument("--token", default=os.environ.get("ROBOT_TOKEN"))
    ap.add_argument("--arm-url", default="http://127.0.0.1:8765")
    ap.add_argument("--arm-yaml", default=str(REPO / "arm" / "config" / "arm.yaml"))
    cam = ap.add_mutually_exclusive_group()
    cam.add_argument("--camera", type=int, default=0, help="wrist camera index (python -m scrub_arm cams)")
    cam.add_argument("--no-camera", action="store_true", help="simulated arm: placeholder frames")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--mute", action="store_true", help="print instead of speaking")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    if not args.site_url or not args.token:
        ap.error("--site-url and --token are required (or CONVEX_SITE_URL / ROBOT_TOKEN)")
    camera = synthetic_camera() if args.no_camera else open_camera(args.camera)
    agent = Agent(FleetClient(args.site_url, args.token), ArmClient(args.arm_url), Recorder(camera, args.fps),
                  Speaker(enabled=not args.mute), Path(args.arm_yaml))
    try:
        agent.run()
    except KeyboardInterrupt:
        if agent.episode:
            agent._finish_episode({})


if __name__ == "__main__":
    main()
