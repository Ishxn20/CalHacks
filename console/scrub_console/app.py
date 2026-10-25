"""Build a Console from settings: arm client, ledger, Claude, Cartesia, Convex, Bitdeer, Droid, audit."""
from __future__ import annotations

import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from scrub_arm.audit import AuditLog
from scrub_arm.config import DEFAULT_CONFIG_DIR, TrayConfig, load_tray_config

from .arm_client import ArmClient
from .bridge import Bridge
from .config import Settings
from .intent import ClaudeParser, Intent, understand
from .ledger import Ledger
from .report import BitdeerClient
from .sinks import ConvexSink, Sink
from .speech import CartesiaSpeech
from .triage import RehearsalRunner, SdkDroidRunner, Triage

log = logging.getLogger(__name__)


class PerceptionCounter:
    """Asks the perception service which instruments the camera sees (POST /count)."""

    def __init__(self, url: str, timeout_s: float = 8.0) -> None:
        self.url, self.timeout_s = url.rstrip("/"), timeout_s

    def __call__(self) -> list[str] | None:
        req = urllib.request.Request(self.url + "/count", data=b"{}", headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            log.warning("perception count unavailable: %s", e)
            return None
        return [i["id"] for i in data.get("instruments", []) if i.get("id")]


@dataclass
class Console:
    settings: Settings
    tray: TrayConfig
    arm: ArmClient
    ledger: Ledger
    bridge: Bridge
    speech: CartesiaSpeech
    audit: AuditLog
    triage: Triage
    bitdeer: BitdeerClient | None
    sinks: list[Sink] = field(default_factory=list)
    parser: ClaudeParser | None = None
    sim: bool = False
    demo: bool = False
    agents: object = None            # AgentNetwork when running with --agents
    dataset: object = None           # DatasetRecorder when running with --record

    def capabilities(self) -> dict:
        runner = self.triage.runner
        return {"sim": self.sim, "demo": self.demo, "claude": self.parser is not None,
                "voice": self.speech.available, "convex": any(isinstance(s, ConvexSink) for s in self.sinks),
                "convex_url": self.settings.convex_url,
                "agents": self.agents.addresses if self.agents else None,
                "bitdeer": self.bitdeer is not None,
                "droid": None if runner is None else ("rehearsal" if getattr(runner, "simulated", False) else "sdk"),
                "recording": None if self.dataset is None else ("simulated" if self.dataset.simulated else "camera")}


def _droid_runner(rehearse: bool):
    if shutil.which("droid"):
        try:
            import droid_sdk  # noqa: F401

            return SdkDroidRunner()
        except ImportError:
            log.warning("droid CLI found but the droid-sdk package is not installed")
    return RehearsalRunner() if rehearse else None


def _dataset(record: str, sinks: list, bridge: Bridge, sim: bool):
    """Record every arm job for the training set (`record` = a camera index, or "sim" for placeholder frames)."""
    convex = next((s for s in sinks if isinstance(s, ConvexSink)), None)
    if convex is None:
        log.warning("--record needs Convex (CONVEX_URL + CONVEX_DEPLOY_KEY) to store episodes; not recording")
        return None
    from .dataset import DatasetRecorder
    from .recorder import Recorder, open_camera, synthetic_camera

    camera = synthetic_camera() if record == "sim" else open_camera(int(record))
    ds = DatasetRecorder(Recorder(camera), convex.client, bridge.emit, simulated=record == "sim")
    bridge.observers.append(ds.observe)
    return ds


def build_console(settings: Settings | None = None, sim: bool = False, demo: bool = False,
                  rehearse_droid: bool = False, arm_config: Path | None = None, agents: bool = False,
                  record: str | None = None) -> Console:
    s = settings or Settings.from_env()
    tray = load_tray_config()
    arm = ArmClient(s.arm_url, s.arm_token)
    ledger = Ledger.from_tray(tray, in_set=not sim)      # the real arm picks from in_set; the sim from tray slots
    audit = AuditLog(s.audit_dir / "console_audit.jsonl")

    has_claude_credentials = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    parser = ClaudeParser(tray, model=s.claude_model) if has_claude_credentials else None
    if parser is None:
        log.info("no Anthropic credentials: requests are understood by the rule-based parser")

    def understand_fn(text: str, context: str) -> Intent:
        return understand(text, arm.resolve, parser, context, tray)

    sinks: list[Sink] = []
    if s.convex_url:
        sinks.append(ConvexSink(s.convex_url, s.convex_deploy_key))

    triage = Triage(arm_config or DEFAULT_CONFIG_DIR / "arm.yaml", _droid_runner(rehearse_droid), s.audit_dir / "triage",
                    audit=audit)
    bitdeer = BitdeerClient(s.bitdeer_key, s.bitdeer_base_url, s.bitdeer_model) if s.bitdeer_key else None
    bridge = Bridge(arm, ledger, understand_fn, sinks=sinks, audit=audit, sim=sim, triage=triage, bitdeer=bitdeer,
                    observe=None if sim else PerceptionCounter(s.perception_url))
    triage.emit = bridge.emit
    dataset = _dataset(record, sinks, bridge, sim) if record else None
    speech = CartesiaSpeech(api_key=os.environ.get("CARTESIA_API_KEY"), voice_id=s.cartesia_voice,
                            tts_model=s.cartesia_tts_model, stt_model=s.cartesia_stt_model)
    net = None
    if agents:
        from .agents import AgentArm, AgentNetwork

        look = bridge.sim_visible if sim else PerceptionCounter(s.perception_url)
        net = AgentNetwork(understand_fn, arm, look=look,
                           on_hop=lambda src, dst, what: bridge.emit("agent_hop", source=src, target=dst, what=what)).start()
        bridge.arm, bridge.understand, bridge.observe = AgentArm(net, arm), net.understand, net.look
    return Console(s, tray, arm, ledger, bridge, speech, audit, triage, bitdeer, sinks, parser, sim, demo, net, dataset)
