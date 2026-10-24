import pytest

from scrub_arm.catalog import Catalog, ResolveError, normalize_orientation, normalize_size, parse_freeform
from scrub_arm.config import load_tray_config


@pytest.fixture
def catalog(config_dir):
    return Catalog(load_tray_config(config_dir / "tray.yaml"))


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
