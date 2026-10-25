"""Settings from the environment (and an optional repo-root .env). Nothing secret is ever read from a tracked file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """KEY=VALUE lines into os.environ, without overriding what is already set. No quoting rules beyond '...' and "..."."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        os.environ.setdefault(k.strip(), v)


@dataclass
class Settings:
    arm_url: str = "http://127.0.0.1:8765"
    arm_token: str | None = None
    perception_url: str = "http://127.0.0.1:8770"
    claude_model: str = "claude-opus-5-5"
    cartesia_voice: str | None = None
    cartesia_tts_model: str = "sonic-3"
    cartesia_stt_model: str = "ink-whisper"
    convex_url: str | None = None
    convex_deploy_key: str | None = None
    bitdeer_base_url: str = "https://api-inference.bitdeer.ai/v1"
    bitdeer_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    bitdeer_key: str | None = None
    audit_dir: Path = ROOT / "logs"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        e = os.environ.get
        return cls(
            arm_url=e("SCRUB_ARM_URL", cls.arm_url), arm_token=e("SCRUB_API_TOKEN"),
            perception_url=e("SCRUB_PERCEPTION_URL", cls.perception_url),
            claude_model=e("SCRUB_CLAUDE_MODEL", cls.claude_model),
            cartesia_voice=e("CARTESIA_VOICE_ID"),
            cartesia_tts_model=e("CARTESIA_TTS_MODEL", cls.cartesia_tts_model),
            cartesia_stt_model=e("CARTESIA_STT_MODEL", cls.cartesia_stt_model),
            convex_url=e("CONVEX_URL"), convex_deploy_key=e("CONVEX_DEPLOY_KEY"),
            bitdeer_base_url=e("BITDEER_BASE_URL", cls.bitdeer_base_url),
            bitdeer_model=e("BITDEER_MODEL", cls.bitdeer_model), bitdeer_key=e("BITDEER_API_KEY"),
            audit_dir=Path(e("SCRUB_LOG_DIR", str(cls.audit_dir))),
        )
