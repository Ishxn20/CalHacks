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
