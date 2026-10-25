"""The scrub assistant on Agentverse, usable from an ASI:One chat.

    python -m scrub_console asi          # needs AGENTVERSE_API_KEY; the console must be running

A uAgent that speaks the Agent Chat Protocol and receives messages through its Agentverse mailbox, so nothing on
the OR laptop listens on the internet. Each chat message is turned into an action on the console (loopback HTTP):

  reads    "what's with the surgeon?", "status", "count", "case report", "what's next?"   answered at once
  motion   "Potts scissors, backhand", "release", "go home"   the console's own parser (Claude) says what it
           would do, and the arm moves only after the user replies "confirm"
  stop     "stop" freezes the arm immediately: stopping is always safe

A chat message comes from the internet, so motion is refused unless the arm is simulated or the sender's address
is in SCRUB_ASI_ALLOWED_SENDERS. Whatever is asked, the arm server still enforces its own safety checks.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

log = logging.getLogger(__name__)

NAME = "Scrub assistant"
MOTION = {"handoff", "stage", "release", "home", "return", "reset", "approve", "reject"}
STATUS = re.compile(r"\b(status|where|what'?s (out|with|on)|with the surgeon|board|tray|holding|doing)\b", re.I)
YES = re.compile(r"^\W*(yes|yep|confirm(ed)?|go( ahead)?|do it|ok(ay)?|proceed)\b", re.I)
NO = re.compile(r"^\W*(no|nope|cancel|never ?mind|don'?t)\b", re.I)
README = """# Scrub assistant

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)
![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)

Talk to a robotic surgical scrub assistant (a 6-axis arm that hands instruments to a surgeon) from ASI:One.
Ask what is on the tray or with the surgeon, run the end-of-case instrument count, get the case report, or
request an instrument. Anything that moves the arm is read back and waits for you to confirm.
"""


class Console:
    """The console's loopback API (scrub_console/server.py)."""

    def __init__(self, url: str = "http://127.0.0.1:8780", timeout_s: float = 30.0) -> None:
        self.url, self.timeout_s = url.rstrip("/"), timeout_s

    def call(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            err = json.loads(e.read() or b"{}")
            raise RuntimeError(err.get("say") or err.get("error") or f"HTTP {e.code}") from e


def describe(state: dict) -> str:
    """The board in two or three sentences."""
    names = lambda status: [i["name"] for i in state["ledger"]["items"] if i["status"] == status]  # noqa: E731
    arm = state.get("arm") or {}
    lines = [f"The arm is {arm.get('phase', 'unknown')}{', holding the ' + arm['holding'] if arm.get('holding') else ''}"
             f"{' (simulated)' if state.get('sim') else ''}."]
    for status, label in (("in_field", "With the surgeon"), ("used", "Used"), ("tray", "On the tray")):
        items = names(status)
        lines.append(f"{label}: {', '.join(items) if items else 'none'}.")
    return " ".join(lines)


def respond(text: str, sender: str, pending: dict, console: Console, allowed: set[str]) -> str:
    """One chat turn. `pending` maps sender -> the request waiting for confirmation (mutated in place)."""
    t = (text or "").strip()
    waiting = pending.pop(sender, None)
    if waiting and YES.match(t):
        return console.call("/api/utterance", {"text": waiting})["say"]
    if waiting and NO.match(t):
        return "Cancelled. Nothing moved."
    if re.fullmatch(r"\W*(stop|freeze|halt)\W*", t, re.I):
        console.call("/api/utterance", {"text": "stop"})
        return "Stopped. The arm is frozen in place, holding whatever it had. Say reset to clear it."
    if STATUS.search(t) and not re.search(r"\b(hand|give|get|stage|need)\b", t, re.I):
        return describe(console.call("/api/state"))
    i = console.call("/api/understand", {"text": t})
    kind = i["intent"]
    if kind == "count":
        return console.call("/api/count", {})["say"]
    if kind == "report":
        return console.call("/api/report", {})["text"]
    if kind == "suggest":
        p = console.call("/api/predict", {})["prediction"]
        return f"Probably the {p['name']} next. {p.get('reason', '')}" if p else "No confident suggestion yet."
    if kind in MOTION:
        if not i.get("sim") and sender not in allowed:
            return "This arm only takes motion requests from its own team's ASI:One accounts. I can still answer questions."
        pending[sender] = t
        return f"{i['say']} Reply \"confirm\" to move the arm, or \"cancel\"."
    if kind == "used":
        return console.call("/api/utterance", {"text": t})["say"]
    return i["say"]                          # a question back (ambiguous request), or the parser's reply


def agent_seed(log_dir: Path) -> str:
    """The agent's identity: SCRUB_AGENT_SEED, else a random seed kept in logs/ (git-ignored) so the address is stable."""
    if os.environ.get("SCRUB_AGENT_SEED"):
        return os.environ["SCRUB_AGENT_SEED"]
    path = log_dir / "asi_agent_seed"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_text().strip()


def run(console_url: str, log_dir: Path, port: int = 8001) -> None:
    import uagents.asgi

    uagents.asgi.HOST = "127.0.0.1"          # messages arrive through the Agentverse mailbox; listen on loopback only
    from uagents import Agent, Context, Protocol
    from uagents_core.contrib.protocols.chat import (ChatAcknowledgement, ChatMessage, EndSessionContent, TextContent,
                                                     chat_protocol_spec)
    from uagents_core.utils.registration import RegistrationRequestCredentials, register_chat_agent

    key = os.environ.get("AGENTVERSE_API_KEY")
    if not key:
        raise SystemExit("set AGENTVERSE_API_KEY (agentverse.ai > Profile > API keys)")
    seed = agent_seed(log_dir)
    agent = Agent(name="scrub-assistant", seed=seed, port=port, mailbox=True)
    console, pending = Console(console_url), {}
    allowed = {a.strip() for a in os.environ.get("SCRUB_ASI_ALLOWED_SENDERS", "").split(",") if a.strip()}
    chat = Protocol(spec=chat_protocol_spec)

    def message(text: str) -> ChatMessage:
        return ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(),
                           content=[TextContent(type="text", text=text), EndSessionContent(type="end-session")])

    @chat.on_message(ChatMessage)
    async def on_chat(ctx: Context, sender: str, msg: ChatMessage):
        await ctx.send(sender, ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id))
        text = " ".join(c.text for c in msg.content if isinstance(c, TextContent)).strip()
        if not text:
            return
        ctx.logger.info(f"from {sender}: {text!r}")
        try:
            reply = respond(text, sender, pending, console, allowed)
        except (OSError, RuntimeError, KeyError) as e:     # console down, arm refused, bad reply
            reply = f"I couldn't do that: {e}"
        await ctx.send(sender, message(reply))

    @chat.on_message(ChatAcknowledgement)
    async def on_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
        pass

    agent.include(chat, publish_manifest=True)
    register_chat_agent(
        NAME, agent.agentverse.mailbox_endpoint, active=True,
        credentials=RegistrationRequestCredentials(agentverse_api_key=key, agent_seed_phrase=seed),
        description="Ask a robotic surgical scrub assistant for instruments, counts and case status.",
        readme=README,
        starter_prompts=["What's with the surgeon right now?", "Run the instrument count", "Hand me the Potts scissors"],
        metadata={"categories": ["healthcare", "robotics"], "is_public": "True"},
    )
    print(f"{NAME} registered on Agentverse as {agent.address}", flush=True)
    agent.run()
