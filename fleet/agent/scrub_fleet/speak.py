"""The arm talks back in the OR through Cartesia: confirmations, warnings, and why it stopped.

A scrub tech's job is half call-and-response ("needle holder, six-oh Prolene, backhand" ->
"needle holder, coming"), so the arm confirms every request out loud and says why it stopped.
Speech runs on a background queue so it never blocks telemetry or recording; a newer urgent
line (a fault) jumps the queue.
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import tempfile
import threading

log = logging.getLogger(__name__)

MODEL = "sonic-3.5"


class Speaker:
    def __init__(self, api_key: str | None = None, voice_id: str | None = None, enabled: bool = True) -> None:
        self.api_key = api_key or os.environ.get("CARTESIA_API_KEY")
        self.voice_id = voice_id or os.environ.get("CARTESIA_VOICE_ID")
        self.enabled = enabled and bool(self.api_key)
        if enabled and not self.api_key:
            log.warning("CARTESIA_API_KEY not set; the arm will print what it would say instead")
        self._q: queue.PriorityQueue[tuple[int, int, str]] = queue.PriorityQueue()
        self._n = 0
        self._client = None
        threading.Thread(target=self._run, name="speaker", daemon=True).start()

    def say(self, text: str, urgent: bool = False) -> None:
        text = text.strip()
        if not text:
            return
        print(f"[arm says] {text}", flush=True)
        if self.enabled:
            self._n += 1
            self._q.put((0 if urgent else 1, self._n, text))

    def _voice(self) -> str:
        if not self.voice_id:
            # No voice configured: use the first voice on the account.
            self.voice_id = next(iter(self._client.voices.list())).id
            log.info("using Cartesia voice %s (set CARTESIA_VOICE_ID to choose)", self.voice_id)
        return self.voice_id

    def _run(self) -> None:
        while True:
            _, _, text = self._q.get()
            try:
                if self._client is None:
                    from cartesia import Cartesia

                    self._client = Cartesia(api_key=self.api_key)
                audio = self._client.tts.generate(
                    model_id=MODEL,
                    transcript=text,
                    voice={"id": self._voice()},
                    language="en",
                    output_format={"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000},
                )
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    path = f.name
                audio.write_to_file(path)
                subprocess.run(["afplay", path] if _is_mac() else ["aplay", "-q", path], check=False, timeout=30)
                os.unlink(path)
            except Exception as e:  # noqa: BLE001 - speech must never take the agent down
                log.warning("speech failed (%s): %s", type(e).__name__, e)


def _is_mac() -> bool:
    import sys

    return sys.platform == "darwin"


def confirm_line(kind: str, arm_reply: dict) -> str:
    """What the arm says when it accepts (or refuses) a remote/voice command."""
    if "error" in arm_reply:
        return arm_reply.get("say") or f"I can't do that: {arm_reply['error']}"
    if kind in ("handoff", "stage"):
        what = arm_reply.get("say") or "instrument"
        return f"{what}, coming." if kind == "handoff" else f"Staging {what}."
    return {"release": "Releasing.", "return": "Putting it back.", "home": "Going home.",
            "stop": "Stopped.", "reset": "Ready."}.get(kind, "")
