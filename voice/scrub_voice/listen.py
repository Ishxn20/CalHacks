"""Speech -> tool code. `python -m scrub_voice.listen` (push-to-talk) or `--text "hand me a scalpel"`.

Transcribes locally with faster-whisper, resolves the phrase against tray.yaml (the one shared
instrument vocabulary), and overwrites last_command.json with the result.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from scrub_arm.catalog import Catalog, ResolveError
from scrub_arm.config import load_tray_config

OUT_PATH = Path(__file__).resolve().parents[2] / "last_command.json"  # repo root, shared with CV/arm
RATE = 16000


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


def transcribe(model, audio, prompt: str) -> str:
    # The vocabulary prompt biases Whisper toward "Castroviejo", "Potts", "DeBakey"...
    segments, _ = model.transcribe(audio, language="en", initial_prompt=prompt)
    return " ".join(s.text.strip() for s in segments).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", help="skip mic + Whisper and encode this phrase")
    ap.add_argument("--model", default="base.en", help="faster-whisper model (try small.en if it mishears)")
    args = ap.parse_args()

    tray = load_tray_config()
    catalog = Catalog(tray)
    model = prompt = None
    if not args.text:
        from faster_whisper import WhisperModel

        model = WhisperModel(args.model, compute_type="int8")
        prompt = "Surgical instruments: " + ", ".join(
            sorted({p for i in tray.instruments.values() for p in (i.name, *i.aliases)}))

    while True:
        err = None
        if args.text:
            text = args.text
        else:
            audio = record()
            # Exactly 0.0 means macOS is feeding us silence (mic permission denied), not a quiet room.
            err = "mic captured pure silence: check macOS mic permission" if not audio.any() else None
            text = "" if err else transcribe(model, audio, prompt)
        cmd = encode(text, catalog) if text else {**encode("", catalog), "error": err or "heard nothing"}
        save(cmd)
        print(json.dumps(cmd, indent=2))
        if args.text:
            return


if __name__ == "__main__":
    main()
