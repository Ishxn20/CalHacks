"""Where console events go. Every event is {"seq", "t", "kind", "data"}.

MemorySink is the in-process stream the dashboard reads (ring buffer + a condition for server-sent events).
ConvexSink mirrors events into a Convex deployment so any number of dashboards update live. A database
outage must never stop the arm, so ConvexSink writes from a background thread and only logs failures.
"""
from __future__ import annotations

import itertools
import json
import logging
import queue
import threading
import time
import urllib.request

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

    def publish(self, kind: str, data: dict) -> dict | None:
        if kind == "metrics":                    # arm readings go to Convex only; they would flood the browser stream
            return None
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


class ConvexHTTP:
    """Runs Convex functions over its HTTP API. The official Python client holds the GIL for a whole round trip
    (~110 ms to the cloud, measured), freezing every thread in the process, including the simulated arm's control
    loop in the demo. urllib releases the GIL while it waits on the socket."""

    def __init__(self, url: str, deploy_key: str | None = None, timeout_s: float = 10.0) -> None:
        self.url, self.deploy_key, self.timeout_s = url.rstrip("/"), deploy_key, timeout_s

    def mutation(self, name: str, args: dict):
        req = urllib.request.Request(self.url + "/api/mutation", method="POST",
                                     data=json.dumps({"path": name, "args": args, "format": "json"}).encode(),
                                     headers={"Content-Type": "application/json"})
        if self.deploy_key:                      # admin auth: the only way to reach internal functions
            req.add_header("Authorization", "Convex " + self.deploy_key)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = json.loads(resp.read())
        if body.get("status") != "success":
            raise RuntimeError(f"{name}: {body.get('errorMessage') or body}")
        return body.get("value")


class ConvexSink(Sink):
    """Mirrors events into Convex over its HTTP API. The functions it calls live in console/convex/."""

    ROUTES = {"ledger": "instruments:sync", "audit": "audit:anchor", "proposal": "proposals:upsert",
              "metrics": "metrics:record"}

    def __init__(self, url: str, deploy_key: str | None = None, client=None) -> None:
        self.client = client or ConvexHTTP(url, deploy_key)
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
