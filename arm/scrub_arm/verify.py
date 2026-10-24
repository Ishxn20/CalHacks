"""Tray-cam verification hook.

The CV service (separate process, owned by the CV teammate) exposes:

    POST <verify.url>   {"expected": "<instrument id>", "slot": "A1"}
    -> 200 {"ok": true, "detected": "<instrument id>", "confidence": 0.93}

`ok` is the service's own verdict; we also reject low confidence. The arm calls
this while holding the instrument up in the camera's view, before the handoff.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import VerifyCfg

log = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    ok: bool
    detected: str | None = None
    confidence: float | None = None
    note: str = ""


class Verifier:
    def __init__(self, cfg: VerifyCfg) -> None:
        self.cfg = cfg

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.url)

    def verify(self, expected_instrument: str, slot_id: str) -> VerifyResult:
        if not self.cfg.url:
            return VerifyResult(ok=True, note="verification disabled")
        body = json.dumps({"expected": expected_instrument, "slot": slot_id}).encode()
        req = urllib.request.Request(self.cfg.url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                data = json.loads(resp.read() or b"{}")
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            if self.cfg.fail_open:
                log.warning("tray-cam unreachable (%s); proceeding without verification", e)
                return VerifyResult(ok=True, note=f"tray-cam unreachable: {e}")
            return VerifyResult(ok=False, note=f"tray-cam unreachable: {e}")

        detected = data.get("detected")
        conf = data.get("confidence")
        ok = bool(data.get("ok", detected == expected_instrument))
        if conf is not None and float(conf) < self.cfg.min_confidence:
            ok = False
        note = "" if ok else f"tray-cam saw {detected or 'nothing'} ({conf})"
        return VerifyResult(ok=ok, detected=detected, confidence=conf, note=note)
