"""Speech in and out through Cartesia: Ink for speech-to-text, Sonic for text-to-speech.

The surgeon's audio reaches us as a WAV file (the dashboard records it in the browser, so any laptop mic
works and nothing else needs installing). Replies are short sentences, so synthesis is a single call and the
most recent phrases are cached: the same readback ("Releasing.") is instant the second time.
"""
from __future__ import annotations

import logging
from collections import OrderedDict

log = logging.getLogger(__name__)

WAV_24K = {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000}


class SpeechUnavailable(RuntimeError):
    """No Cartesia key / SDK / voice: the console keeps working with text input and on-screen replies."""


class CartesiaSpeech:
    def __init__(self, api_key: str | None = None, voice_id: str | None = None, tts_model: str = "sonic-3",
                 stt_model: str = "ink-whisper", client=None, cache_size: int = 64) -> None:
        self.api_key, self.voice_id = api_key, voice_id
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

    def synthesize(self, text: str) -> bytes:
        """WAV bytes (16-bit PCM, 24 kHz) for `text`."""
        text = text.strip()[:500]
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]
        audio = self.client.tts.generate(model_id=self.tts_model, transcript=text,
                                         voice={"id": self._voice()}, output_format=WAV_24K).read()
        self._cache[text] = audio
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return audio
