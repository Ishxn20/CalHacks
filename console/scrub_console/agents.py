"""The scrub assistant as three Fetch.ai uAgents that really carry the traffic.

    voice agent  : turns a transcript into an intent            (Claude, via scrub_console.intent)
    arm agent    : the only thing that sends motion commands    (the arm's HTTP API)
    eyes agent   : looks at the tray and reports what it sees   (the perception service)

The console talks to them through a small gateway agent, so every utterance, arm command and camera count
is an actual uAgents message between agents, not a function call. `AgentNetwork.understand`, `.look` and
`.arm_call` block until the reply arrives; `AgentArm` makes the arm agent look like an ArmClient to the bridge.
Status polling and the /resolve dry run stay direct HTTP (read-only, and polled several times a second).

Everything runs in one local Bureau. Two choices keep it local:
  - uAgents binds its HTTP endpoint to 0.0.0.0 by default; we set it to 127.0.0.1 before anything starts.
  - A no-op registration policy means nothing is registered on the Almanac or contacted over the network.
To be discoverable on Agentverse instead, drop the policy and pass `mailbox=True` / `agentverse=` to the agents.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import json
import logging
import queue
import socket
import threading
import uuid
from typing import Callable

import uagents.asgi

uagents.asgi.HOST = "127.0.0.1"          # must happen before an Agent or Bureau serves anything

from uagents import Agent, Bureau, Context, Model  # noqa: E402
from uagents_core.registration import BatchRegistrationPolicy  # noqa: E402

from .arm_client import ArmClient, ArmError  # noqa: E402
from .intent import Intent  # noqa: E402

log = logging.getLogger(__name__)


class _LocalOnly(BatchRegistrationPolicy):
    async def register(self):
        pass

    def add_agent(self, agent_info, identity):
        pass


class Understand(Model):
    id: str
    text: str
    context: str


class Understood(Model):
    id: str
    kind: str
    say: str
    fields_json: str
    source: str


class ArmCall(Model):
    id: str
    name: str
    payload_json: str


class ArmResult(Model):
    id: str
    ok: bool
    data_json: str
    error: str
    say: str
    status: int


class Look(Model):
    id: str


class Seen(Model):
    id: str
    ok: bool
    instruments: list[str]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _agent(name: str, loop) -> Agent:
    return Agent(name=name, seed=f"scrub-assistant/{name}", loop=loop, enable_agent_inspector=False,
                 publish_agent_details=False, report_events=False)


class AgentNetwork:
    def __init__(self, understand: Callable[[str, str], Intent], arm: ArmClient,
                 look: Callable[[], list[str] | None] | None = None,
                 on_hop: Callable[[str, str, str], None] | None = None, timeout_s: float = 30.0) -> None:
        self._understand, self._arm, self._look = understand, arm, look
        self._on_hop = on_hop or (lambda src, dst, what: None)
        self.timeout_s = timeout_s
        self._pending: dict[str, cf.Future] = {}
        self._outbox: queue.Queue = queue.Queue()
        self._started = threading.Event()
        loop = asyncio.new_event_loop()
        self.voice, self.arm_agent, self.eyes, self.gateway = (_agent(n, loop) for n in ("voice", "arm", "eyes", "gateway"))
        self._wire()
        self.bureau = Bureau(port=_free_port(), registration_policy=_LocalOnly(), loop=loop, log_level="WARNING")
        for a in (self.voice, self.arm_agent, self.eyes, self.gateway):
            self.bureau.add(a)

    @property
    def addresses(self) -> dict[str, str]:
        return {"voice": self.voice.address, "arm": self.arm_agent.address, "eyes": self.eyes.address}

    # ── the agents ────────────────────────────────────────────────────────

    def _wire(self) -> None:
        net = self

        async def blocking(fn, *a):
            return await asyncio.get_running_loop().run_in_executor(None, fn, *a)     # keep the event loop free

        @self.voice.on_message(model=Understand)
        async def understand(ctx: Context, sender: str, msg: Understand):
            net._on_hop("gateway", "voice", "Understand")
            i = await blocking(net._understand, msg.text, msg.context)
            net._on_hop("voice", "gateway", i.kind)
            await ctx.send(sender, Understood(id=msg.id, kind=i.kind, say=i.say, fields_json=json.dumps(i.fields), source=i.source))

        @self.arm_agent.on_message(model=ArmCall)
        async def arm_call(ctx: Context, sender: str, msg: ArmCall):
            net._on_hop("gateway", "arm", msg.name)

            def call():
                payload = json.loads(msg.payload_json)
                return net._arm.command(msg.name) if msg.name in ("release", "stop", "home", "return", "reset") \
                    else getattr(net._arm, msg.name)(payload)

            try:
                out = await blocking(call)
                reply = ArmResult(id=msg.id, ok=True, data_json=json.dumps(out), error="", say="", status=200)
            except ArmError as e:
                reply = ArmResult(id=msg.id, ok=False, data_json="{}", error=str(e), say=e.say, status=e.status or 0)
            net._on_hop("arm", "gateway", "ok" if reply.ok else "refused")
            await ctx.send(sender, reply)

        @self.eyes.on_message(model=Look)
        async def look(ctx: Context, sender: str, msg: Look):
            net._on_hop("gateway", "eyes", "Look")
            seen = await blocking(net._look) if net._look else None
            net._on_hop("eyes", "gateway", f"{len(seen or [])} instruments")
            await ctx.send(sender, Seen(id=msg.id, ok=seen is not None, instruments=list(seen or [])))

        targets = {Understand: self.voice, ArmCall: self.arm_agent, Look: self.eyes}

        @self.gateway.on_interval(period=0.02)
        async def drain(ctx: Context):
            while True:
                try:
                    msg = net._outbox.get_nowait()
                except queue.Empty:
                    return
                await ctx.send(targets[type(msg)].address, msg)

        def resolver(model):
            @self.gateway.on_message(model=model)
            async def reply(ctx: Context, sender: str, msg):
                fut = net._pending.pop(msg.id, None)
                if fut and not fut.done():
                    fut.set_result(msg)
            return reply

        for m in (Understood, ArmResult, Seen):
            resolver(m)

    # ── lifecycle and blocking calls ──────────────────────────────────────

    def start(self, wait_s: float = 10.0) -> "AgentNetwork":
        threading.Thread(target=self.bureau.run, name="uagents", daemon=True).start()
        # ready once a round trip works
        deadline = threading.Event()
        for _ in range(int(wait_s / 0.2)):
            try:
                self._ask(Look(id=""), 1.0)
                return self
            except cf.TimeoutError:
                deadline.wait(0.2)
        raise RuntimeError("the agent network did not start")

    def _ask(self, msg, timeout_s: float):
        msg.id = uuid.uuid4().hex
        fut: cf.Future = cf.Future()
        self._pending[msg.id] = fut
        self._outbox.put(msg)
        try:
            return fut.result(timeout=timeout_s)
        except cf.TimeoutError:
            self._pending.pop(msg.id, None)
            raise

    def understand(self, text: str, context: str) -> Intent:
        r: Understood = self._ask(Understand(id="", text=text, context=context), self.timeout_s)
        return Intent(r.kind, r.say, json.loads(r.fields_json), r.source)

    def arm_call(self, name: str, payload: dict | None = None) -> dict:
        r: ArmResult = self._ask(ArmCall(id="", name=name, payload_json=json.dumps(payload or {})), self.timeout_s)
        if not r.ok:
            raise ArmError(r.error, r.say, r.status or None)
        return json.loads(r.data_json)

    def look(self) -> list[str] | None:
        r: Seen = self._ask(Look(id=""), self.timeout_s)
        return r.instruments if r.ok else None


class AgentArm:
    """Looks like an ArmClient to the bridge; motion commands go through the arm agent, reads stay direct."""

    def __init__(self, net: AgentNetwork, direct: ArmClient) -> None:
        self.net, self.direct = net, direct
        self.status, self.audit, self.resolve = direct.status, direct.audit, direct.resolve

    def handoff(self, fields: dict) -> dict:
        return self.net.arm_call("handoff", fields)

    def stage(self, fields: dict) -> dict:
        return self.net.arm_call("stage", fields)

    def command(self, name: str) -> dict:
        return self.net.arm_call(name)
