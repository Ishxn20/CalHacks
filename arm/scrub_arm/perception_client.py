"""Client for the perception service's /plan endpoint (perception/scrub_vision/service.py)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .handoff import TaskFailed

ACTIONS = ("grasp_target", "move_aside", "fail")


class PerceptionClient:
    def __init__(self, url: str, timeout_s: float = 5.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def plan(self, instrument_id: str) -> dict:
        """POST /plan; TaskFailed (with a speakable message) if the service is down or answers nonsense."""
        req = urllib.request.Request(self.url + "/plan", data=json.dumps({"instrument": instrument_id}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read() or b"{}")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise TaskFailed(f"perception unavailable: {e}") from e
        if not isinstance(data, dict) or data.get("action") not in ACTIONS:
            raise TaskFailed(f"perception gave a bad plan: {data!r}"[:200])
        if data["action"] != "fail" and not isinstance(data.get("target"), dict):
            raise TaskFailed("perception gave a plan without a target")
        if data["action"] == "move_aside" and not isinstance(data.get("place"), dict):
            raise TaskFailed("perception gave a move_aside without a place")
        return data
