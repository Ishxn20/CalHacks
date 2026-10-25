import pytest

from scrub_console.wake import WakeGate


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


@pytest.fixture
def gate():
    clock = Clock()
    g = WakeGate(follow_up_s=8.0, clock=clock)
    g.clock_ = clock
    return g


@pytest.mark.parametrize("heard, wanted", [
    ("Scrub, Potts scissors", "Potts scissors"),
    ("scrub castroviejo backhand", "castroviejo backhand"),
    ("Hey Scrub. Count check.", "Count check"),
    ("Scrubs, what's next?", "what's next"),
    ("Scrub.", ""),
])
def test_the_wake_word_is_stripped_and_the_rest_is_acted_on(gate, heard, wanted):
    assert gate.check(heard) == wanted


@pytest.mark.parametrize("chatter", [
    "can you pass me the suction",
    "I think the scrub nurse is on break",       # wake word not at the start
    "she said to stop by later today",            # long sentence that happens to contain "stop"
    "",
])
def test_operating_room_talk_is_ignored(gate, chatter):
    assert gate.check(chatter) is None


def test_a_short_stop_never_needs_the_wake_word(gate):
    assert gate.check("Stop!") == "Stop!"
    assert gate.check("freeze") == "freeze"


def test_got_it_works_without_the_wake_word_only_while_the_arm_holds_an_instrument_out(gate):
    assert gate.check("got it", phase="idle") is None
    assert gate.check("got it", phase="presenting") == "got it"


def test_after_the_robot_speaks_a_follow_up_needs_no_wake_word_for_a_few_seconds(gate):
    gate.open()
    gate.clock_.t += 5
    assert gate.check("approve") == "approve"
    gate.clock_.t += 4
    assert gate.check("approve") is None
