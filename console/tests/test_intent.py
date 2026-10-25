from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from scrub_arm.catalog import Catalog, ResolveError
from scrub_arm.config import load_tray_config
from scrub_console.arm_client import ArmError
from scrub_console.intent import TOOLS, ClaudeParser, rules_parse, understand


@pytest.fixture(scope="module")
def tray():
    return load_tray_config()


@pytest.fixture
def resolve(tray):
    """What the arm's /resolve does in ik mode: the real catalog, errors turned into speakable ArmErrors."""
    catalog = Catalog(tray, require_slot=False)

    def _resolve(fields):
        try:
            r = catalog.resolve(fields.get("instrument", ""), fields.get("suture"), fields.get("size"),
                                fields.get("orientation"))
        except ResolveError as e:
            raise ArmError(str(e), str(e).capitalize(), 400) from e
        return {"instrument": r.instrument.id, "orientation": r.orientation, "say": r.describe()}

    return _resolve


class FakeClaude:
    def __init__(self, tool=None, name=None, stop_reason="tool_use", error=None):
        self.calls, self._error = [], error
        block = [SimpleNamespace(type="tool_use", name=name, input=tool)] if name else [SimpleNamespace(type="text", text="hi")]
        self._resp = SimpleNamespace(stop_reason=stop_reason, content=block, stop_details=SimpleNamespace(category="cyber"))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self._error:
            raise self._error
        return self._resp


def parser(tray, **kw):
    fake = FakeClaude(**kw)
    return ClaudeParser(tray, client=fake), fake


REQ = {"mode": "handoff", "instrument": "the resano", "suture": None, "size": None, "orientation": "backhand"}


def test_claude_request_becomes_a_validated_handoff(tray, resolve):
    p, fake = parser(tray, name="request_instrument", tool={**REQ, "instrument": "the castro"})
    i = understand("hand me the castro backhand", resolve, p, context="idle")
    assert (i.kind, i.source) == ("handoff", "claude")
    assert i.fields == {"instrument": "the castro", "orientation": "backhand"}      # nulls dropped
    assert "Castroviejo" in i.say and "backhand" in i.say


def test_the_request_to_claude_follows_the_model_rules(tray, resolve):
    p, fake = parser(tray, name="request_instrument", tool=REQ)
    understand("resano", resolve, p, context="holding nothing")
    kw = fake.calls[0]
    assert "tool_choice" not in kw                         # forced tool use is a 400 on this model
    assert kw["fallbacks"] == "default" and "server-side-fallback-2026-07-01" in kw["betas"]
    assert kw["output_config"] == {"effort": "low"}
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(kw)
    assert kw["model"] == "claude-opus-5-5" and "Context: holding nothing" in kw["messages"][0]["content"]
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in kw["tools"])


def test_a_hallucinated_instrument_becomes_a_question_not_a_motion(tray, resolve):
    p, _ = parser(tray, name="request_instrument", tool={**REQ, "instrument": "hammer"})
    i = understand("hand me a hammer", resolve, p)
    assert i.kind == "ask" and "hammer" in i.say.lower()


def test_a_category_asks_which_one(tray, resolve):
    p, _ = parser(tray, name="request_instrument", tool={**REQ, "instrument": "scissors", "orientation": None})
    i = understand("scissors please", resolve, p)
    assert i.kind == "ask" and "which scissors" in i.say.lower()


def test_controls_used_and_count(tray, resolve):
    i = understand("got it", resolve, parser(tray, name="control", tool={"command": "release"})[0])
    assert i.kind == "release"
    i = understand("put it back", resolve, parser(tray, name="control", tool={"command": "return_held"})[0])
    assert i.kind == "return"
    i = understand("done with resano", resolve, parser(tray, name="report_used", tool={"instrument": "resano"})[0])
    assert (i.kind, i.fields) == ("used", {"instrument": "forceps_resano"})
    assert understand("count", resolve, parser(tray, name="count_check", tool={})[0]).kind == "count"


def test_claude_asking_a_question_is_passed_through(tray, resolve):
    p, _ = parser(tray, name="ask", tool={"question": "Which needle holder: Castroviejo or Ryder?"})
    i = understand("needle holder", resolve, p)
    assert i.kind == "ask" and i.say.startswith("Which needle holder")


def test_a_refusal_falls_back_to_the_rules(tray, resolve):
    p, _ = parser(tray, name="request_instrument", tool=REQ, stop_reason="refusal")
    i = understand("resano", resolve, p)
    assert (i.kind, i.source) == ("handoff", "rules")


def test_api_errors_and_missing_tool_calls_fall_back_to_the_rules(tray, resolve):
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))
    for p in (parser(tray, error=err)[0], parser(tray, error=RuntimeError("no key"))[0], parser(tray)[0]):
        assert understand("stop", resolve, p).kind == "stop"


def test_an_unknown_command_from_claude_is_a_question(tray, resolve):
    p, _ = parser(tray, name="control", tool={"command": "self_destruct"})
    assert understand("boom", resolve, p).kind == "ask"


@pytest.mark.parametrize("text,kind", [
    ("stop", "stop"), ("please halt", "stop"), ("go home", "home"), ("put it back", "return"),
    ("I've got it", "release"), ("count check", "count"), ("reset", "reset"),
    ("approve", "approve"), ("approve the change", "approve"), ("reject it", "reject"), ("don't apply that", "reject"),
    ("write the case report", "report"), ("what's next", "suggest"),
])
def test_rules_controls(text, kind):
    assert rules_parse(text)[0] == kind


def test_rules_handoff_and_used(resolve):
    i = understand("castro, backhand", resolve)
    assert (i.kind, i.source, i.fields) == ("handoff", "rules", {"instrument": "castro", "orientation": "backhand"})
    i = understand("I'm done with the resano", resolve)
    assert (i.kind, i.fields["instrument"]) == ("used", "forceps_resano")
    assert understand("stage the potts scissors", resolve).kind == "stage"


def test_nothing_heard_and_unknown_words(resolve):
    assert understand("   ", resolve).kind == "ask"
    assert understand("hand me a banana", resolve).kind == "ask"


def test_tool_schemas_are_wellformed():
    names = [t["name"] for t in TOOLS]
    assert names == ["request_instrument", "control", "report_used", "count_check", "decide_fix", "after_case", "ask"]
    for t in TOOLS:
        props = set(t["input_schema"]["properties"])
        assert set(t["input_schema"]["required"]) <= props
        if t["name"] != "count_check":
            assert set(t["input_schema"]["required"]) == props      # strict mode: optional means nullable


@pytest.mark.parametrize("tool, inp, kind", [
    ("decide_fix", {"decision": "approve"}, "approve"), ("decide_fix", {"decision": "reject"}, "reject"),
    ("after_case", {"action": "report"}, "report"), ("after_case", {"action": "suggest_next"}, "suggest"),
])
def test_claude_can_decide_a_fix_and_run_the_after_case_tools(resolve, tool, inp, kind):
    p = ClaudeParser.__new__(ClaudeParser)
    p.call = lambda text, context: (tool, inp)
    assert understand("anything", resolve, p).kind == kind


def test_rules_keep_suture_size_and_orientation_from_a_plain_sentence(tray, resolve):
    kind, fields = rules_parse("hand me the needle holder with 6-0 prolene backhand please", tray.sutures)
    assert kind == "handoff"
    assert (fields["suture"], fields["size"], fields["orientation"]) == ("prolene", "6-0", "backhand")
    assert "prolene" not in fields["instrument"] and "backhand" not in fields["instrument"]
    i = understand("the ryder with polypropylene 4-0", resolve, None, tray=tray)
    assert i.kind == "handoff" and i.fields["suture"] == "prolene" and i.fields["size"] == "4-0"


def test_claude_gets_a_cached_system_prompt(tray, resolve):
    calls = []

    class Fake:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    calls.append(kw)
                    return SimpleNamespace(stop_reason="end_turn", usage=None, content=[
                        SimpleNamespace(type="tool_use", name="count_check", input={})])

    p = ClaudeParser(tray, client=Fake())
    assert understand("count check please", resolve, p).kind == "count"
    system = calls[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"} and "scrub assistant" in system[0]["text"]
