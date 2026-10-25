"""Speech -> tool code. `python -m scrub_voice.listen` (push-to-talk) or `--text "hand me a scalpel"`.

Transcribes with Cartesia Ink (the same speech-to-text the console uses; needs CARTESIA_API_KEY, read from
the environment or the repo-root .env), resolves the phrase against tray.yaml (the one shared instrument
vocabulary), and overwrites last_command.json with the result.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import time
import wave
from pathlib import Path

from scrub_arm.catalog import Catalog, ResolveError
from scrub_arm.config import load_tray_config

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "last_command.json"  # repo root, shared with CV/arm
RATE = 16000
STT_MODEL = "ink-whisper"


def encode(text: str, catalog: Catalog) -> dict:
    out = {"ok": False, "heard": text, "instrument": None, "category": None, "slot": None,
           "orientation": None, "error": None, "ts": time.time()}
    try:
        r = catalog.resolve(text)
    except ResolveError as e:
        out["error"] = str(e)
        return out
    out.update(ok=True, instrument=r.instrument.id, category=r.instrument.category,
               slot=r.slot.id, orientation=r.orientation)
    return out


def save(cmd: dict, path: Path = OUT_PATH) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(cmd, f, indent=2)
    os.replace(tmp, path)  # readers never see a half-written file


def record():
    import numpy as np
    import sounddevice as sd

    chunks = []
    input("Press Enter to start talking...")
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32",
                        callback=lambda data, *_: chunks.append(data.copy())):
        input("Listening... press Enter to stop.")
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, dtype="float32")


def to_wav(audio, rate: int = RATE) -> bytes:
    """float32 samples in [-1, 1] -> 16-bit mono WAV bytes (what Cartesia's batch speech-to-text accepts)."""
    import numpy as np

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def cartesia_client(api_key: str | None = None):
    key = api_key or os.environ.get("CARTESIA_API_KEY") or _dotenv("CARTESIA_API_KEY")
    if not key:
        raise SystemExit("set CARTESIA_API_KEY (environment or repo-root .env) for speech-to-text")
    from cartesia import Cartesia

    return Cartesia(api_key=key)


def _dotenv(name: str) -> str | None:
    env = ROOT / ".env"
    if not env.exists():
        return None
    for line in env.read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == name and v.strip():
            return v.strip().strip("'\"")
    return None


def transcribe(client, audio) -> str:
    """Cartesia Ink. There is no vocabulary prompt like Whisper's; the catalog's aliases absorb near misses."""
    resp = client.stt.transcribe(file=("speech.wav", to_wav(audio), "audio/wav"), model=STT_MODEL, language="en")
    return (resp.text or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", help="skip mic + speech-to-text and encode this phrase")
    args = ap.parse_args()

    catalog = Catalog(load_tray_config())
    client = None if args.text else cartesia_client()

    while True:
        err = None
        if args.text:
            text = args.text
        else:
            audio = record()
            # Exactly 0.0 means macOS is feeding us silence (mic permission denied), not a quiet room.
            err = "mic captured pure silence: check macOS mic permission" if not audio.any() else None
            text = "" if err else transcribe(client, audio)
        cmd = encode(text, catalog) if text else {**encode("", catalog), "error": err or "heard nothing"}
        save(cmd)
        print(json.dumps(cmd, indent=2))
        if args.text:
            return


if __name__ == "__main__":
    main()
