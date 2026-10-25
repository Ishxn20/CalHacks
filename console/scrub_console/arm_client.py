"""Client for the arm's HTTP API (arm/scrub_arm/server.py). Errors carry the `say` text meant for speech."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class ArmError(RuntimeError):
    def __init__(self, message: str, say: str | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.say, self.status = say or message, status


class ArmClient:
    def __init__(self, url: str, token: str | None = None, timeout_s: float = 5.0) -> None:
        self.url, self.token, self.timeout_s = url.rstrip("/"), token, timeout_s

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        if self.token:
            req.add_header("Authorization", "Bearer " + self.token)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read() or b"{}")
            except ValueError:
                err = {}
            raise ArmError(err.get("error") or f"HTTP {e.code}", err.get("say"), e.code) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ArmError(f"arm unreachable: {e}", "I can't reach the arm.") from e

    def status(self, since: float | None = None) -> dict:
        return self._call("GET", "/status" if since is None else f"/status?since={since:.3f}")

    def audit(self) -> dict:
        return self._call("GET", "/audit")

    def resolve(self, fields: dict) -> dict:
        return self._call("POST", "/resolve", fields)

    def handoff(self, fields: dict) -> dict:
        return self._call("POST", "/handoff", fields)

    def stage(self, fields: dict) -> dict:
        return self._call("POST", "/stage", fields)

    def command(self, name: str) -> dict:
        """release | stop | home | return | reset"""
        return self._call("POST", "/" + name, {})
