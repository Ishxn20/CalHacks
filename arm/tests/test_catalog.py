import pytest

from scrub_arm.catalog import Catalog, ResolveError, normalize_orientation, normalize_size, parse_freeform
from scrub_arm.config import load_tray_config


@pytest.fixture
def tray(config_dir):
    return load_tray_config(config_dir / "tray.yaml")


@pytest.fixture
def catalog(tray):
    return Catalog(tray)


def test_generic_needle_holder_matches_suture_loaded_slot(catalog):
    r = catalog.resolve("Needle holder", "Prolene", "6-0", "forehand")
    assert r.slot.id == "A1"
    assert r.orientation == "forehand"
    r = catalog.resolve("needle holder", "polypropylene", "4/0", "backwards")
    assert r.slot.id == "A2"
    assert r.orientation == "backhand"


def test_specific_model_and_spoken_size(catalog):
    assert catalog.resolve("castro", size="six-oh").slot.id == "A1"


def test_unavailable_suture_is_an_error_not_a_guess(catalog):
    with pytest.raises(ResolveError, match="have 4-0 prolene, 6-0 prolene"):
        catalog.resolve("needle holder", "vicryl")
    with pytest.raises(ResolveError, match="no 7-0"):
        catalog.resolve("needle holder", "prolene", "7-0")


def test_blade_numbers_never_fuzzy_match_each_other(catalog):
    assert catalog.resolve("11 blade").slot.id == "A4"
    assert catalog.resolve("number eleven blade".replace("eleven", "11")).slot.id == "A4"
    with pytest.raises(ResolveError, match="no #10 scalpel"):
        catalog.resolve("10 blade")


def test_phrase_inside_sentence_and_typos(catalog):
    assert catalog.resolve("give me the potts scissors please").slot.id == "B1"
    assert catalog.resolve("debakey forcep").slot.id == "A3"
    assert catalog.resolve("cross clamp").slot.id == "B2"


def test_orientation_only_for_orientable_instruments(catalog):
    assert catalog.resolve("debakey", orientation="backhand").orientation is None
    assert catalog.resolve("potts", orientation="back").orientation == "backhand"
    assert catalog.resolve("potts").orientation == "forehand"


def test_unknown_instrument(catalog):
    with pytest.raises(ResolveError):
        catalog.resolve("hammer")


@pytest.mark.parametrize("raw,want", [
    ("6-0", "6-0"), ("6/0", "6-0"), ("6 0", "6-0"), ("60", "6-0"), ("six oh", "6-0"),
    ("10-0", "10-0"), ("size 4-0", "4-0"), ("1.5 mm", "1.5mm"), ("1.5", "1.5mm"), (None, None),
])
def test_normalize_size(raw, want):
    assert normalize_size(raw) == want


def test_normalize_orientation():
    assert normalize_orientation("Fore-hand") == "forehand"
    assert normalize_orientation("back hand") == "backhand"
    with pytest.raises(ResolveError):
        normalize_orientation("sideways")


def test_parse_freeform(catalog):
    f = parse_freeform("Needle holder, 6-0 Prolene, backhand")
    assert f == {"instrument": "Needle holder", "size": "6-0", "suture": "Prolene", "orientation": "backhand"}
    assert catalog.resolve(**f).slot.id == "A1"


def test_resolve_without_slot(tray):
    c = Catalog(tray, require_slot=False)
    r = c.resolve("resano")
    assert r.slot is None and r.instrument.id == "forceps_resano"
    assert "Resano" in r.describe()


def test_resolve_with_slot_still_rejects_unloaded(tray):
    with pytest.raises(ResolveError):
        Catalog(tray).resolve("resano")


def test_resolve_without_slot_keeps_orientation(tray):
    r = Catalog(tray, require_slot=False).resolve("castro", orientation="backhand")
    assert r.orientation == "backhand"


def test_category_request_in_ik_mode_asks_which(tray):
    with pytest.raises(ResolveError, match="which"):
        Catalog(tray, require_slot=False).resolve("scissors")


def test_a_named_instrument_beats_a_longer_category_word(catalog):
    # "rosano" (how speech-to-text hears Resano) is shorter than "forceps"; it must still win, so the request
    # is refused (no Resano on the demo tray) rather than quietly becoming the DeBakey forceps.
    with pytest.raises(ResolveError, match="Resano"):
        catalog.resolve("Rosano Forceps.")
    assert catalog.resolve("forceps").instrument.id == "forceps_debakey"     # a bare category still works
