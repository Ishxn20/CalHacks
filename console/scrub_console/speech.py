"""Speech in and out through Cartesia: Ink for speech-to-text, Sonic for text-to-speech.

The surgeon's audio reaches us as a WAV file (the dashboard records it in the browser, so any laptop mic
works and nothing else needs installing). Replies are short sentences, so synthesis is a single call and the
most recent phrases are cached: the same readback ("Releasing.") is instant the second time.

Two Sonic features make it sound like a scrub tech rather than a screen reader:
  - a pronunciation dictionary for instrument and suture names it otherwise mangles (see PRONUNCIATIONS);
  - generation_config: routine readbacks are calm and quick, alerts (a fault, a failed pick) firmer and louder.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path

log = logging.getLogger(__name__)

WAV_24K = {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000}

ROUTINE = {"emotion": "confident", "speed": 1.05}
URGENT = {"emotion": "determined", "speed": 1.1, "volume": 1.4}

# Words Sonic gets wrong by default, with respellings checked by round trip (synthesize, then transcribe with
# Ink): without them it says "castroviho" and "vicarol". Only add a word after the same check: a respelling can
# easily make things worse ("DeBakey" is right by default; "Debaykee" comes out as "Dubai key").
PRONUNCIATIONS = [
    {"text": "Castroviejo", "alias": "Castroveeayho"},
    {"text": "Vicryl", "alias": "Vycril"},
]


class SpeechUnavailable(RuntimeError):
    """No Cartesia key / SDK / voice: the console keeps working with text input and on-screen replies."""


class CartesiaSpeech:
    def __init__(self, api_key: str | None = None, voice_id: str | None = None, tts_model: str = "sonic-3",
                 stt_model: str = "ink-whisper", client=None, cache_size: int = 64,
                 state_dir: Path | None = None, pronunciation_dict_id: str | None = None) -> None:
        self.api_key, self.voice_id = api_key, voice_id
        self.state_dir, self.dict_id = state_dir, pronunciation_dict_id
        self._dict_tried = bool(pronunciation_dict_id)
        self.tts_model, self.stt_model = tts_model, stt_model
        self._client = client
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._cache_size = cache_size

    @property
    def available(self) -> bool:
        return self._client is not None or bool(self.api_key)

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise SpeechUnavailable("set CARTESIA_API_KEY to enable voice")
            try:
                from cartesia import Cartesia
            except ImportError as e:
                raise SpeechUnavailable("install the voice extra: pip install -e 'console[voice]'") from e
            self._client = Cartesia(api_key=self.api_key)
        return self._client

    def transcribe(self, wav: bytes, language: str = "en") -> str:
        resp = self.client.stt.transcribe(file=("speech.wav", wav, "audio/wav"), model=self.stt_model, language=language)
        return (resp.text or "").strip()

    def _voice(self) -> str:
        if not self.voice_id:
            # first item only: list() would auto-paginate through the whole voice catalogue (minutes)
            voice = next(iter(self.client.voices.list(limit=1)), None)
            if voice is None:
                raise SpeechUnavailable("no Cartesia voices available: set CARTESIA_VOICE_ID")
            self.voice_id = voice.id
            log.info("using Cartesia voice %s (%s); set CARTESIA_VOICE_ID to choose", voice.id, getattr(voice, "name", "?"))
        return self.voice_id

    def pronunciation_dict(self) -> str | None:
        """Id of a Cartesia pronunciation dictionary holding PRONUNCIATIONS, created once and remembered in
        state_dir (a new one is made only when the word list changes). None if it can't be had: speech still works."""
        if self._dict_tried:
            return self.dict_id
        self._dict_tried = True
        digest = hashlib.sha256(json.dumps(PRONUNCIATIONS, sort_keys=True).encode()).hexdigest()[:12]
        cache = self.state_dir / "cartesia_pronunciation.json" if self.state_dir else None
        try:
            if cache and cache.exists():
                saved = json.loads(cache.read_text())
                if saved.get("hash") == digest:
                    self.dict_id = saved["id"]
                    return self.dict_id
            created = self.client.pronunciation_dicts.create(name=f"scrub-or-{digest}", items=PRONUNCIATIONS)
            self.dict_id = created.id
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"hash": digest, "id": self.dict_id}))
            log.info("Cartesia pronunciation dictionary %s (%d words)", self.dict_id, len(PRONUNCIATIONS))
        except Exception as e:  # noqa: BLE001 - pronunciation is a nicety; never lose speech over it
            log.warning("no Cartesia pronunciation dictionary (%s); speaking without it", e)
            self.dict_id = None
        return self.dict_id

    def synthesize(self, text: str, urgent: bool = False) -> bytes:
        """WAV bytes (16-bit PCM, 24 kHz) for `text`; `urgent` for alerts (firmer, louder)."""
        text = text.strip()[:500]
        key = ("!" if urgent else "") + text
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        extra = {"pronunciation_dict_id": d} if (d := self.pronunciation_dict()) else {}
        audio = self.client.tts.generate(model_id=self.tts_model, transcript=text, voice={"id": self._voice()},
                                         output_format=WAV_24K, generation_config=URGENT if urgent else ROUTINE,
                                         **extra).read()
        self._cache[key] = audio
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return audio
