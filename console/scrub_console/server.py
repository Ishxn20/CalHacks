"""The console's own HTTP server: serves the dashboard and its API on loopback.

    GET  /                        the dashboard
    GET  /api/state               everything the dashboard shows + which integrations are live
    GET  /api/events?since=N      server-sent events (every console event, newest after N)
    GET  /api/arm                 joint angles vs setpoints and the collision-guard limit (the gauges)
    POST /api/audio  (WAV body)   hands-free speech, transcribed by Cartesia, acted on only after the wake word
    POST /api/heard {text}        hands-free speech already transcribed in the browser (no Cartesia key), same gate
    POST /api/utterance {text}    a request with no wake word needed (CLI, tests, the Fetch.ai agent)
    POST /api/understand {text}   dry run: the intent and reply a request would get, without acting on it
    GET  /api/tts?text=...        the reply as speech (Cartesia)
    POST /api/count               count check               POST /api/used {instrument}
    POST /api/report              end-of-case report        POST /api/predict   next-instrument suggestion
    POST /api/triage/approve|reject {id}
    GET  /api/audit               audit chains: arm + console
    simulation only:  POST /api/sim/remove {instrument} | /api/sim/restore
    --demo only:      POST /api/triage/trigger {fault} | /api/demo/tamper

It never leaves loopback and never sends CORS headers: the Host must be a loopback name and a browser
request must come from this page's own origin, so another site cannot drive the console (and through it the arm).
"""
from __future__ import annotations

import json
import logging
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml
from scrub_arm.audit import verify

from .app import Console
from .arm_client import ArmError
from .speech import SpeechUnavailable

log = logging.getLogger(__name__)

WEB = Path(__file__).resolve().parent / "web"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MAX_BODY = 8 * 1024 * 1024


def _guard_deg(arm_yaml: Path) -> float | None:
    """The collision guard: a joint lagging its setpoint by more than this freezes the arm."""
    try:
        return (yaml.safe_load(arm_yaml.read_text()).get("motion") or {}).get("tracking_error_limit_deg")
    except (OSError, yaml.YAMLError, AttributeError):
        return None


def _hostname(h: str) -> str:
    h = h.strip()
    if h.startswith("["):
        return h[1:].split("]", 1)[0]
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def make_handler(console: Console):
    bridge = console.bridge

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            log.debug("%s - %s", self.address_string(), fmt % args)

        # ── helpers ────────────────────────────────────────────────────────
        def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, default=str).encode())

        def _fail(self, code: int, error: str, say: str | None = None) -> None:
            self._json(code, {"error": error, "say": say or error})

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                raise ValueError("body too large")
            return self.rfile.read(n) if n else b""

        def _args(self) -> dict:
            raw = self._body()
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                raise ValueError("body must be a JSON object")
            return data

        def _same_origin(self) -> bool:
            if _hostname(self.headers.get("Host") or "") not in LOOPBACK:
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin.rstrip("/") == f"http://{self.headers.get('Host')}"

        # ── routing ────────────────────────────────────────────────────────
        def do_GET(self):
            if not self._same_origin():
                return self._fail(403, "request refused")
            url = urlparse(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
                if url.path == "/favicon.ico":
                    return self._send(204, b"")
                if url.path == "/vendor/convex-browser.js":
                    return self._send(200, (WEB / "vendor" / "convex-browser.js").read_bytes(), "text/javascript; charset=utf-8")
                if url.path == "/api/state":
                    return self._json(200, {**bridge.snapshot(), "capabilities": console.capabilities()})
                if url.path == "/api/arm":                  # joint gauges, polled ~2 Hz: the cached /status, no arm call
                    arm = bridge.status.get("arm") or {}
                    return self._json(200, {"phase": bridge.status.get("phase"), "q_deg": arm.get("q_deg") or [],
                                            "q_cmd_deg": arm.get("q_cmd_deg") or [], "gripper_deg": arm.get("gripper_deg"),
                                            "guard_deg": _guard_deg(console.triage.config_path)})
                if url.path == "/api/events":
                    return self._events(int((parse_qs(url.query).get("since") or ["0"])[0]))
                if url.path == "/api/tts":
                    text = (parse_qs(url.query).get("text") or [""])[0]
                    return self._send(200, console.speech.synthesize(text), "audio/wav")
                if url.path == "/api/audit":
                    return self._json(200, self._audit())
                self._fail(404, "not found")
            except SpeechUnavailable as e:
                self._fail(503, str(e), "Voice isn't set up.")
            except Exception as e:  # noqa: BLE001
                log.exception("GET %s failed", url.path)
                self._fail(500, f"{type(e).__name__}: {e}")

        def do_POST(self):
            if not self._same_origin():
                return self._fail(403, "request refused")
            path = urlparse(self.path).path
            try:
                if path == "/api/audio":
                    return self._json(200, self._voice(self._body()))
                a = self._args()
                if path == "/api/heard":            # hands-free: browser speech recognition, gated by the wake word
                    return self._json(200, bridge.handle_heard(str(a.get("text", ""))[:500], via="speech"))
                if path == "/api/utterance":
                    return self._json(200, bridge.handle_text(str(a.get("text", ""))[:500]))
                if path == "/api/understand":       # dry run: what a request would do, nothing moves (ASI:One agent)
                    i = bridge.understand(str(a.get("text", ""))[:500], bridge.context())
                    return self._json(200, {"intent": i.kind, "say": i.say, "source": i.source, "sim": bridge.sim})
                if path == "/api/count":
                    return self._json(200, bridge.count_check())
                if path == "/api/used":
                    return self._json(200, bridge.handle_text("done with " + str(a.get("instrument", ""))))
                if path == "/api/report":
                    return self._json(200, bridge.case_report())
                if path == "/api/predict":
                    return self._json(200, bridge.suggest_next())
                if path == "/api/triage/approve":
                    return self._json(200, console.triage.approve(str(a.get("id")), who="dashboard").public())
                if path == "/api/triage/reject":
                    return self._json(200, console.triage.reject(str(a.get("id")), who="dashboard").public())
                if path == "/api/sim/remove":
                    return self._json(200, {"removed": bridge.sim_remove(str(a.get("instrument", "")))})
                if path == "/api/sim/restore":
                    bridge.sim_restore()
                    return self._json(200, {"restored": True})
                if path == "/api/triage/trigger" and console.demo:
                    fault = str(a.get("fault") or "joint3 is 28.0 deg off its setpoint (collision or blocked?) - arm frozen in place")
                    bridge.emit("fault", message=fault, simulated=True)
                    return self._json(200, {"id": console.triage.trigger(fault, bridge.recent_context(), simulated=True)})
                if path == "/api/demo/tamper" and console.demo:
                    return self._json(200, self._tamper())
                self._fail(404, "not found")
            except SpeechUnavailable as e:
                self._fail(503, str(e), "Voice isn't set up.")
            except PermissionError as e:
                self._fail(403, str(e))
            except (KeyError, ValueError) as e:
                self._fail(400, str(e).strip("'\""))
            except ArmError as e:
                self._fail(502, str(e), e.say)
            except Exception as e:  # noqa: BLE001
                log.exception("POST %s failed", path)
                self._fail(500, f"{type(e).__name__}: {e}")

        # ── handlers ───────────────────────────────────────────────────────
        def _events(self, since: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            seq = since
            try:
                while True:
                    events = bridge.stream.wait(seq, 15.0)
                    if not events:
                        self.wfile.write(b": ping\n\n")
                    for e in events:
                        seq = e["seq"]
                        self.wfile.write(f"id: {seq}\ndata: {json.dumps(e, default=str)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        def _voice(self, wav: bytes) -> dict:
            if not wav:
                raise ValueError("no audio received")
            return bridge.handle_heard(console.speech.transcribe(wav), via="voice")   # silence/chatter -> ignored

        def _audit(self) -> dict:
            return bridge.anchor()

        def _tamper(self) -> dict:
            """Non-destructive: edit a COPY of the console audit log and show that verification catches it."""
            src = console.audit.path
            copy = src.with_name("audit.tampered-demo.jsonl")
            lines = src.read_text().splitlines()
            if len(lines) < 2:
                raise ValueError("do something first (say a request), then tamper with the record of it")
            i = len(lines) // 2
            lines[i] = lines[i].replace('"kind": "', '"kind": "X', 1)
            copy.write_text("\n".join(lines) + "\n")
            v = verify(copy)
            bridge.emit("tamper_demo", ok=v.ok, error=v.error, line=i + 1)
            return {"detected": not v.ok, "error": v.error, "edited_line": i + 1, "original_intact": console.audit.verify().ok}

    return Handler


def serve(console: Console, host: str = "127.0.0.1", port: int = 8780) -> ThreadingHTTPServer:
    if host not in LOOPBACK:
        raise ValueError("the console only listens on loopback (127.0.0.1 / localhost / ::1)")
    httpd = ThreadingHTTPServer((host, port), make_handler(console))
    httpd.daemon_threads = True
    return httpd
