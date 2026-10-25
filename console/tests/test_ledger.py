import pytest

from scrub_console.ledger import HELD, IN_FIELD, TRAY, USED, Ledger, UnknownInstrument, load_groups


def make(*ids, groups=None):
    return Ledger(ids, {i: i.replace("_", " ").title() for i in ids}, groups or {})


def test_everything_starts_on_the_tray_and_the_count_is_clean():
    lg = make("forceps_resano", "scissors_potts")
    assert lg.counts() == {TRAY: 2, HELD: 0, IN_FIELD: 0, USED: 0}
    r = lg.reconcile(["forceps_resano", "scissors_potts"])
    assert r.ok and r.checked_visually and "All 2 instruments" in lg.say_for(r)


def test_handing_over_and_getting_it_back_in_the_used_zone():
    lg = make("forceps_resano", "scissors_potts")
    lg.move("forceps_resano", HELD)
    lg.move("forceps_resano", IN_FIELD)
    assert lg.counts()[IN_FIELD] == 1
    r = lg.reconcile(["scissors_potts"])
    assert not r.ok and r.outstanding == ["Forceps Resano"] and r.missing == []
    lg.move("forceps_resano", USED)
    assert lg.reconcile(["forceps_resano", "scissors_potts"]).ok


def test_the_demo_moment_a_removed_instrument_is_named():
    lg = make("forceps_resano", "scissors_potts", "clamp_side_biting")
    r = lg.reconcile(["forceps_resano", "scissors_potts"])
    assert not r.ok and r.missing == ["Clamp Side Biting"]
    assert "Missing: Clamp Side Biting" in lg.say_for(r)


def test_something_extra_on_the_table_is_flagged():
    lg = make("forceps_resano")
    r = lg.reconcile(["forceps_resano", "scissors_potts"])
    assert not r.ok and r.unexpected == ["scissors potts"]


def test_interchangeable_instruments_satisfy_each_other():
    lg = make("scalpel_11", groups={"scalpel_10": "scalpel", "scalpel_11": "scalpel"})
    assert lg.reconcile(["scalpel_10"]).ok


def test_without_a_visual_count_only_outstanding_instruments_matter():
    lg = make("forceps_resano")
    assert lg.reconcile().ok and not lg.reconcile().checked_visually
    lg.move("forceps_resano", HELD)
    assert not lg.reconcile().ok


def test_duplicate_copies_are_tracked_separately():
    lg = make("scalpel_11", "scalpel_11")
    lg.move("scalpel_11", HELD)
    assert lg.counts()[TRAY] == 1 and lg.counts()[HELD] == 1
    assert [i.uid for i in lg.items] == ["scalpel_11#1", "scalpel_11#2"]


def test_moves_only_leave_states_that_make_sense():
    lg = make("forceps_resano")
    assert lg.move("forceps_resano", USED) is None          # nothing is in the field to return
    assert lg.move("forceps_resano", HELD) is not None
    assert lg.move("forceps_resano", HELD) is None          # already held
    with pytest.raises(UnknownInstrument):
        lg.move("hammer", HELD)


def test_a_failed_pick_returns_the_instrument_to_the_tray():
    lg = make("forceps_resano")
    lg.move("forceps_resano", HELD)
    assert lg.move("forceps_resano", TRAY) is not None and lg.counts()[TRAY] == 1


def test_groups_come_from_the_perception_config():
    g = load_groups()
    assert g["scalpel_10"] == g["scalpel_20"] == "scalpel"
    assert "scissors_micro_forward" not in g                 # the two micro scissors are told apart now


def test_the_real_arm_board_is_the_in_set():
    from scrub_arm.config import load_tray_config

    tray = load_tray_config()
    assert [i["instrument"] for i in Ledger.from_tray(tray, in_set=True).snapshot()] == list(tray.in_set)
    assert len(Ledger.from_tray(tray).snapshot()) == len(tray.slots)


def test_from_tray_makes_one_item_per_slot():
    from scrub_arm.config import load_tray_config

    tray = load_tray_config()
    lg = Ledger.from_tray(tray)
    assert len(lg.items) == len(tray.slots) == 8
    assert lg.name("forceps_resano") == "Resano forceps"
