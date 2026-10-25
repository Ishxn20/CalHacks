"""Where console events go. Every event is {"seq", "t", "kind", "data"}.

MemorySink is the in-process stream the dashboard reads (ring buffer + a condition for server-sent events).
ConvexSink mirrors events into a Convex deployment so any number of dashboards update live. A database
outage must never stop the arm, so ConvexSink writes from a background thread and only logs failures.
"""
from __future__ import annotations

import itertools
import logging
import queue
import threading
import time

log = logging.getLogger(__name__)


class Sink:
    def publish(self, kind: str, data: dict) -> None:
        raise NotImplementedError


class MemorySink(Sink):
    def __init__(self, maxlen: int = 500) -> None:
        self.maxlen = maxlen
        self.events: list[dict] = []
        self._seq = 0
        self._cond = threading.Condition()

    def publish(self, kind: str, data: dict) -> dict:
        with self._cond:
            self._seq += 1
            event = {"seq": self._seq, "t": round(time.time(), 3), "kind": kind, "data": data}
            self.events.append(event)
            del self.events[:-self.maxlen]
            self._cond.notify_all()
            return event

    def since(self, seq: int = 0) -> list[dict]:
        with self._cond:
            return [e for e in self.events if e["seq"] > seq]

    def wait(self, seq: int, timeout: float = 15.0) -> list[dict]:
        """Events after `seq`, blocking up to `timeout` seconds if there are none yet."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq > seq, timeout)
            return [e for e in self.events if e["seq"] > seq]

    def of_kind(self, *kinds: str) -> list[dict]:
        with self._cond:
            return [e for e in self.events if e["kind"] in kinds]


class ConvexSink(Sink):
    """Mirrors events into Convex via the official Python client. The functions it calls live in console/convex/."""

    ROUTES = {"ledger": "instruments:sync", "audit": "audit:anchor", "proposal": "proposals:upsert"}

    def __init__(self, url: str, deploy_key: str | None = None, client=None) -> None:
        if client is None:
            from convex import ConvexClient

            client = ConvexClient(url)
            if deploy_key:
                client.set_admin_auth(deploy_key)
        self.client = client
        self.failures = 0
        self._q: queue.Queue = queue.Queue(maxsize=1000)
        self._counter = itertools.count(1)
        threading.Thread(target=self._run, name="convex-sink", daemon=True).start()

    def publish(self, kind: str, data: dict) -> None:
        try:
            self._q.put_nowait((kind, data, next(self._counter), time.time()))
        except queue.Full:
            self.failures += 1
            log.warning("convex queue full; dropping %s event", kind)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            kind, data, seq, t = item
            name = self.ROUTES.get(kind, "events:add")
            args = {"kind": kind, "data": data, "seq": seq, "t": t} if name == "events:add" else data
            try:
                self.client.mutation(name, args)
            except Exception as e:  # noqa: BLE001 - a database problem must not reach the arm
                self.failures += 1
                log.warning("convex %s failed: %s", name, e)
            finally:
                self._q.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        done = threading.Event()
        threading.Thread(target=lambda: (self._q.join(), done.set()), daemon=True).start()
        return done.wait(timeout)
