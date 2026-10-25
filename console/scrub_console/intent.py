"""Spoken request -> one validated intent. Claude does the understanding; the arm's own catalog has the last word.

Claude is given a fixed set of tools and must answer with exactly one of them. Whatever it picks, an
instrument request is only accepted if the arm's /resolve dry run accepts it, so a hallucinated instrument
can never become a motion: it turns into a spoken question instead. If Claude is unreachable, refuses, or
no key is set, a small rule-based parser (plus the catalog's own free-form parser) takes over, so the demo
keeps working offline.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from scrub_arm.catalog import ORIENTATIONS as ORIENTATION_WORDS, parse_freeform
from scrub_arm.config import TrayConfig

from .arm_client import ArmError

log = logging.getLogger(__name__)

CONTROLS = ("release", "stop", "home", "return_held", "reset")
ORIENTATIONS = ["forehand", "backhand"]


@dataclass
class Intent:
    kind: str                 # handoff | stage | release | stop | home | return | reset | used | count | ask
    say: str
    fields: dict = field(default_factory=dict)
    source: str = "rules"     # claude | rules


Resolve = Callable[[dict], dict]


def _nullable(t: dict) -> dict:
    return {**t, "type": [t["type"], "null"]}


TOOLS = [
    {"name": "request_instrument", "strict": True,
     "description": "The surgeon asks for an instrument to be picked and handed over (or pre-staged).",
     "input_schema": {"type": "object", "additionalProperties": False,
                      "required": ["mode", "instrument", "suture", "size", "orientation"],
                      "properties": {
                          "mode": {"type": "string", "enum": ["handoff", "stage"],
                                   "description": "stage only if they say to get it ready or hold it"},
                          "instrument": {"type": "string", "description": "as the surgeon said it"},
                          "suture": _nullable({"type": "string", "description": "e.g. prolene, vicryl"}),
                          "size": _nullable({"type": "string", "description": "e.g. 6-0, 4-0, 1.5 mm"}),
                          "orientation": {"type": ["string", "null"], "enum": ORIENTATIONS + [None]}}}},
    {"name": "control", "strict": True,
     "description": "A command to the arm itself: release (surgeon has the instrument), stop, go home, "
                    "put the held instrument back, or clear a stop/fault.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["command"],
                      "properties": {"command": {"type": "string", "enum": list(CONTROLS)}}}},
    {"name": "report_used", "strict": True,
     "description": "The surgeon says they are finished with an instrument and it goes to the used zone.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["instrument"],
                      "properties": {"instrument": {"type": "string"}}}},
    {"name": "count_check", "strict": True,
     "description": "The surgeon asks for an instrument count / count check.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": [], "properties": {}}},
    {"name": "ask", "strict": True,
     "description": "The request is ambiguous, incomplete or not understood. Ask ONE short question.",
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["question"],
                      "properties": {"question": {"type": "string"}}}},
]


def system_prompt(tray: TrayConfig) -> str:
    vocab = "\n".join(f"- {i.name}: " + ", ".join(i.aliases) if i.aliases else f"- {i.name}"
                      for i in tray.instruments.values())
    return (
        "You are the voice interface of a robotic scrub assistant (a demo, not a medical device). A surgeon "
        "speaks short requests in a noisy operating room; speech-to-text may mishear words. Turn each request "
        "into exactly one tool call.\n\n"
        "Instruments you can fetch (name: things people call it):\n" + vocab + "\n\n"
        "Rules:\n"
        "- Always answer by calling exactly one tool. Never reply with plain text.\n"
        "- Only request instruments from the list. If the name is a category (\"scissors\", \"forceps\") or "
        "could be two instruments, call ask with one short question naming the choices.\n"
        "- Do not guess a suture, size or orientation that was not said; use null.\n"
        "- \"Context\" tells you what the arm is doing now; use it for requests like \"flip it\" or \"I've got it\".\n"
        "- If you cannot tell what was asked, call ask."
    )


class ClaudeParser:
    """Wraps one Messages API call. `client` can be any object with the SDK's `beta.messages.create`."""

    def __init__(self, tray: TrayConfig, client=None, model: str = "claude-opus-5-5") -> None:
        self.system, self.model = system_prompt(tray), model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def call(self, text: str, context: str) -> tuple[str, dict] | None:
        """(tool name, tool input) chosen by Claude, or None on a refusal / no tool call / API trouble."""
        try:
            import anthropic
        except ImportError:
            log.warning("anthropic is not installed; using rules")
            return None
        try:
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=4096, system=self.system, tools=TOOLS,
                output_config={"effort": "low"},               # a classification: keep latency down
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                messages=[{"role": "user", "content": f"Context: {context}\nSurgeon said: {text}"}],
            )
        except anthropic.RateLimitError:
            log.warning("claude rate limited; using rules")
        except anthropic.APIConnectionError as e:
            log.warning("claude unreachable (%s); using rules", e)
        except anthropic.APIStatusError as e:
            log.warning("claude API error %s; using rules", e.status_code)
        except Exception as e:  # noqa: BLE001 - missing key, bad install: never block the demo
            log.warning("claude unavailable (%s); using rules", e)
        else:
            if resp.stop_reason == "refusal":
                log.warning("claude declined (%s); using rules", getattr(resp.stop_details, "category", None))
                return None
            for block in resp.content:
                if block.type == "tool_use":
                    return block.name, dict(block.input)
            log.warning("claude made no tool call (stop_reason=%s); using rules", resp.stop_reason)
        return None


# ── rule-based fallback ────────────────────────────────────────────────────

_RULES = [
    ("stop", r"\b(stop|halt|freeze|abort)\b"),
    ("reset", r"\b(reset|clear (the )?(fault|stop))\b"),
    ("count", r"\b(count|how many|accounted)\b"),
    ("home", r"\b(go home|home position|move home)\b"),
    ("return", r"\b(put (it|that) back|return (it|that))\b"),
    ("release", r"\b(release|i('ve| have) (got )?it|got it|take it)\b"),
]
_USED = re.compile(r"\b(?:done with|finished with|finished|used|i'm done with)\s+(?:the\s+)?(.+)", re.I)


_SIZE = re.compile(r"\b\d{1,2}\s*[-/]\s*0\b|\b\d(?:\.\d)?\s?mm\b")


def _extract_details(t: str, sutures: dict[str, list[str]]) -> tuple[str, dict]:
    """Pull suture / size / orientation out of a free-form phrase so they are not silently dropped."""
    fields: dict = {}
    for canon, aliases in sutures.items():
        for a in sorted([canon, *aliases], key=len, reverse=True):
            if re.search(rf"\b{re.escape(a)}\b", t):
                fields["suture"] = canon
                t = re.sub(rf"\b{re.escape(a)}\b", " ", t)
                break
    m = _SIZE.search(t)
    if m:
        fields["size"] = m.group()
        t = t.replace(m.group(), " ")
    for word in sorted(ORIENTATION_WORDS, key=len, reverse=True):
        if re.search(rf"\b{word}\b", t):
            fields["orientation"] = word
            t = re.sub(rf"\b{word}\b", " ", t)
            break
    return re.sub(r"\s+", " ", t).strip(" ,."), fields


def rules_parse(text: str, sutures: dict[str, list[str]] | None = None) -> tuple[str, dict]:
    t = text.lower().strip()
    for kind, pattern in _RULES:
        if re.search(pattern, t):
            return kind, {}
    m = _USED.search(t)
    if m:
        return "used", {"instrument": m.group(1).strip(" .")}
    kind = "handoff"
    if re.match(r"(stage|get ready|hold)\b", t):
        kind, t = "stage", re.sub(r"^(stage|get ready|hold)( the)?\s*", "", t)
    if "," in t:                                     # "needle holder, 6-0 prolene, backhand"
        return kind, parse_freeform(t)
    rest, fields = _extract_details(t, sutures or {})
    return kind, {"instrument": rest, **fields}


# ── entry point ────────────────────────────────────────────────────────────

KINDS = {"handoff", "stage", "used", "ask", "release", "stop", "home", "return", "reset", "count"}
_SAY = {"release": "Releasing.", "stop": "Stopping.", "home": "Going home.", "return": "Putting it back.",
        "reset": "Cleared.", "count": "Counting now."}


def _clean(fields: dict) -> dict:
    return {k: v for k, v in fields.items() if v not in (None, "")}


def understand(text: str, resolve: Resolve, parser: ClaudeParser | None = None, context: str = "idle",
               tray: TrayConfig | None = None) -> Intent:
    """Turn a transcript into an Intent. Never raises: problems become an `ask` intent with a speakable message."""
    text = (text or "").strip()
    if not text:
        return Intent("ask", "I didn't hear anything.")
    source, picked = "rules", None
    if parser is not None:
        picked = parser.call(text, context)
        source = "claude" if picked else "rules"
    kind, fields = _from_tool(*picked) if picked else rules_parse(text, tray.sutures if tray else None)
    if kind not in KINDS:
        return Intent("ask", "Sorry, say that again?", source=source)

    if kind == "ask":
        return Intent("ask", fields.get("question") or "Sorry, say that again?", source=source)
    if kind in _SAY:
        return Intent(kind, _SAY[kind], source=source)
    try:
        info = resolve(_clean(fields) if kind != "used" else {"instrument": fields.get("instrument", "")})
    except ArmError as e:
        return Intent("ask", e.say, source=source)
    if kind == "used":
        return Intent("used", f"Noted, {info['say']} is done.", {"instrument": info["instrument"]}, source)
    verb = "Staging" if kind == "stage" else "On it:"
    return Intent(kind, f"{verb} {info['say']}.", _clean(fields), source)


def _from_tool(name: str, inp: dict) -> tuple[str, dict]:
    if name == "request_instrument":
        return inp.get("mode", "handoff"), {k: inp.get(k) for k in ("instrument", "suture", "size", "orientation")}
    if name == "control":
        cmd = inp.get("command")
        return {"return_held": "return"}.get(cmd, cmd), {}
    if name == "report_used":
        return "used", {"instrument": inp.get("instrument", "")}
    if name == "count_check":
        return "count", {}
    return "ask", {"question": inp.get("question")}
