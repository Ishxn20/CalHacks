import json

import pytest

from scrub_arm.catalog import Catalog
from scrub_arm.config import load_tray_config
from scrub_voice.listen import encode, save


@pytest.fixture
def catalog():
    return Catalog(load_tray_config())


def test_generic_phrase_encodes_to_instrument_id(catalog):
    r = encode("hand me a scalpel", catalog)
    assert (r["ok"], r["instrument"], r["category"], r["slot"]) == (True, "scalpel_11", "scalpel", "A4")


def test_blade_number_is_exact(catalog):
    # 15 blade isn't on the demo tray, so it must fail rather than fall back to the 11.
    assert encode("can I get the 11 blade", catalog)["instrument"] == "scalpel_11"
    assert encode("can I get the 15 blade", catalog)["ok"] is False


def test_unknown_phrase_is_not_ok(catalog):
    r = encode("hand me a banana", catalog)
    assert r["ok"] is False and r["instrument"] is None and r["error"]


def test_save_roundtrip(catalog, tmp_path):
    cmd = encode("hand me a scalpel", catalog)
    save(cmd, tmp_path / "last_command.json")
    assert json.loads((tmp_path / "last_command.json").read_text()) == cmd


def test_audio_is_sent_to_cartesia_ink_as_16_bit_wav():
    import io
    import wave
    from types import SimpleNamespace

    import numpy as np

    from scrub_voice.listen import RATE, transcribe

    calls = []
    client = SimpleNamespace(stt=SimpleNamespace(
        transcribe=lambda **kw: calls.append(kw) or SimpleNamespace(text="  potts scissors  ")))
    audio = np.sin(np.linspace(0, 200, RATE)).astype("float32")          # 1 s tone
    assert transcribe(client, audio) == "potts scissors"
    name, wav, mime = calls[0]["file"]
    assert (name, mime, calls[0]["model"], calls[0]["language"]) == ("speech.wav", "audio/wav", "ink-whisper", "en")
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, RATE, RATE)


def test_the_key_can_come_from_the_repo_env_file(tmp_path, monkeypatch):
    import scrub_voice.listen as listen

    monkeypatch.setattr(listen, "ROOT", tmp_path)
    (tmp_path / ".env").write_text("OTHER=x\nCARTESIA_API_KEY='sk_test'\n")
    assert listen._dotenv("CARTESIA_API_KEY") == "sk_test"
    assert listen._dotenv("MISSING") is None
