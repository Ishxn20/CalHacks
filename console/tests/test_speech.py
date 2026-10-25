from types import SimpleNamespace

import pytest

from scrub_console.speech import CartesiaSpeech, SpeechUnavailable


class FakeCartesia:
    def __init__(self, voices=("v1",)):
        self.tts_calls, self.stt_calls, self._voices = [], [], voices
        self.stt = SimpleNamespace(transcribe=self._transcribe)
        self.tts = SimpleNamespace(generate=self._generate)
        self.voices = SimpleNamespace(list=self._list)

    def _list(self, limit=None):
        """Like the SDK: an auto-paginating iterator. Walking past the first page would mean paging the whole catalogue."""
        for v in self._voices:
            yield SimpleNamespace(id=v, name="n")
        if self._voices:
            raise AssertionError("paged past the first voice")

    def _transcribe(self, **kw):
        self.stt_calls.append(kw)
        return SimpleNamespace(text="  castroviejo backhand  ")

    def _generate(self, **kw):
        self.tts_calls.append(kw)
        return SimpleNamespace(read=lambda: b"RIFFdata")


def test_transcribe_sends_a_wav_to_ink_and_trims_the_text():
    fake = FakeCartesia()
    text = CartesiaSpeech(client=fake).transcribe(b"WAVBYTES")
    assert text == "castroviejo backhand"
    kw = fake.stt_calls[0]
    assert kw["model"] == "ink-whisper" and kw["language"] == "en"
    assert kw["file"] == ("speech.wav", b"WAVBYTES", "audio/wav")


def test_synthesize_uses_sonic_wav_output_and_caches():
    fake = FakeCartesia()
    s = CartesiaSpeech(client=fake, voice_id="abc")
    assert s.synthesize("Releasing.") == b"RIFFdata"
    assert s.synthesize("Releasing.") == b"RIFFdata"
    assert len(fake.tts_calls) == 1                                  # second one came from the cache
    kw = fake.tts_calls[0]
    assert kw["model_id"] == "sonic-3" and kw["transcript"] == "Releasing." and kw["voice"] == {"id": "abc"}
    assert kw["output_format"] == {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000}


def test_without_a_configured_voice_the_first_account_voice_is_used():
    fake = FakeCartesia(voices=("first", "second"))
    s = CartesiaSpeech(client=fake)
    s.synthesize("hello")
    assert fake.tts_calls[0]["voice"] == {"id": "first"} and s.voice_id == "first"


def test_the_cache_is_bounded():
    s = CartesiaSpeech(client=FakeCartesia(), voice_id="v", cache_size=2)
    for t in ("a", "b", "c"):
        s.synthesize(t)
    assert list(s._cache) == ["b", "c"]


def test_missing_key_is_a_clear_error_not_a_crash():
    s = CartesiaSpeech(api_key=None)
    assert not s.available
    with pytest.raises(SpeechUnavailable, match="CARTESIA_API_KEY"):
        s.synthesize("hello")
    with pytest.raises(SpeechUnavailable):
        CartesiaSpeech(client=FakeCartesia(voices=()), voice_id=None).synthesize("hello")


def test_real_sdk_accepts_the_arguments_we_send():
    """The installed SDK's signatures, not our fake: the keyword names must exist."""
    import inspect

    from cartesia import Cartesia

    c = Cartesia(api_key="x")
    assert {"model_id", "transcript", "voice", "output_format"} <= set(inspect.signature(c.tts.generate).parameters)
    assert {"file", "model", "language"} <= set(inspect.signature(c.stt.transcribe).parameters)
