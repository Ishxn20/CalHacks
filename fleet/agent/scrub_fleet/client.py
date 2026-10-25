"""HTTP clients: the fleet cloud (Convex HTTP actions) and the local arm server (scrub_arm serve)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class HttpError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


def _request(url: str, data: bytes | None = None, headers: dict | None = None, method: str | None = None,
             timeout: float = 10.0) -> bytes:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, e.read().decode(errors="replace")) from e


class FleetClient:
    """Talks to convex/http.ts with this robot's token."""

    def __init__(self, site_url: str, token: str, timeout: float = 10.0) -> None:
        self.base = site_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _post(self, path: str, body: dict | None = None) -> dict:
        raw = _request(self.base + path, json.dumps(body or {}).encode(),
                       {"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"},
                       timeout=self.timeout)
        return json.loads(raw or b"{}")

    def heartbeat(self, **body) -> dict:
        return self._post("/robot/heartbeat", body)

    def commands(self) -> list[dict]:
        return self._post("/robot/commands")["commands"]

    def command_result(self, command_id: str, ok: bool, result: str) -> None:
        self._post("/robot/command-result", {"commandId": command_id, "ok": ok, "result": result})

    def start_episode(self, job_id: int, kind: str, request: str | None, started_at_ms: float) -> dict:
        return self._post("/robot/episodes/start",
                          {"jobId": job_id, "kind": kind, "request": request, "startedAt": started_at_ms})

    def upload(self, data: bytes, content_type: str) -> str:
        """Store a file in Convex storage; returns its storage id."""
        url = self._post("/robot/upload-url")["url"]
        raw = _request(url, data, {"Content-Type": content_type}, method="POST", timeout=120)
        return json.loads(raw)["storageId"]

    def finish_episode(self, **body) -> None:
        self._post("/robot/episodes/finish", body)

    def incident(self, fault: str, phase: str, context: dict) -> str:
        return self._post("/robot/incidents", {"fault": fault, "phase": phase, "context": context})["incidentId"]


class ArmClient:
    """The arm's own HTTP API (arm/scrub_arm/server.py); it keeps every safety check of its own."""

    def __init__(self, url: str = "http://127.0.0.1:8765", timeout: float = 5.0) -> None:
        self.base = url.rstrip("/")
        self.timeout = timeout

    def status(self) -> dict:
        return json.loads(_request(self.base + "/status", timeout=self.timeout))

    def post(self, path: str, body: dict | None = None) -> dict:
        try:
            raw = _request(self.base + path, json.dumps(body or {}).encode(),
                           {"Content-Type": "application/json"}, timeout=self.timeout)
        except HttpError as e:
            try:
                reply = json.loads(e.body)
            except ValueError:
                reply = {}
            return {"error": reply.get("error", str(e)), "say": reply.get("say"), "status": e.status}
        return json.loads(raw or b"{}")
