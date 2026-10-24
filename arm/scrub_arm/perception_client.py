"""Client for the perception service's /plan endpoint (perception/scrub_vision/service.py)."""
from __future__ import annotations

import json
import math
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
        if data["action"] != "fail":
            _numbers(data.get("target"), ("x_mm", "y_mm", "yaw_deg", "thickness_mm"), "target")
        if data["action"] == "move_aside":
            _numbers(data.get("place"), ("x_mm", "y_mm", "yaw_deg"), "place")
        return data


def _numbers(d, keys, what) -> None:
    if not isinstance(d, dict):
        raise TaskFailed(f"perception gave a plan without a {what}")
    for k in keys:
        v = d.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise TaskFailed(f"perception gave a bad {what} {k}: {v!r}")
