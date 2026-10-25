"""HTTP API for the voice pipeline, CV/prediction service, and demo UI.

    POST /handoff   {"instrument": "needle holder", "suture": "prolene", "size": "6-0", "orientation": "backhand"}
                    or {"text": "needle holder, 6-0 prolene, backhand"} (fallback if the LLM parser is down)
    POST /stage     same body - pick the predicted next instrument and hold it near the surgeon
    POST /resolve   same body - dry run: which slot would be used? (no motion)
    POST /release   surgeon has it: open the jaws (also accepted mid-swing; latched)
    POST /return    put whatever is held back in its slot
    POST /home
    POST /stop      freeze immediately (motors stay energized, nothing drops)
    POST /reset     clear a stop/fault so jobs are accepted again
    GET  /status
    GET  /audit     is the audit log's hash chain intact? {"ok", "entries", "head", "error"}

Errors come back as {"error": ..., "say": ...}; `say` is short enough to feed straight to TTS.

Access control (this server moves a real arm):
  - token: when set, every request except the CORS preflight needs `Authorization: Bearer <token>`.
  - Browsers: a request carrying an Origin header is refused unless that origin is in `allowed_origins`,
    and CORS headers are only sent to allowed origins. Without this any web page open on the same
    machine could POST /handoff. Clients that are not browsers send no Origin and are unaffected.
  - Host: when bound to loopback the Host header must be a loopback name (stops DNS rebinding).
  - audit: every state-changing request and every refusal is appended to the hash-chained audit log.
"""
from __future__ import annotations

import hmac
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .audit import AuditLog
from .catalog import Catalog, ResolveError, parse_freeform
from .handoff import Busy, ScrubAssistant

log = logging.getLogger(__name__)

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})
LOGGED_FIELDS = ("instrument", "suture", "size", "orientation", "text")


def _hostname(host_header: str) -> str:
    h = host_header.strip()
    if h.startswith("["):
        return h[1:].split("]", 1)[0]
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


def make_handler(assistant: ScrubAssistant, catalog: Catalog, token: str | None = None,
                 allowed_origins=(), audit: AuditLog | None = None, enforce_host: bool = False):
    origins = frozenset(o.rstrip("/") for o in allowed_origins)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # route through logging instead of stderr
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _cors(self) -> None:
            origin = self.headers.get("Origin")
            if origin and origin.rstrip("/") in origins:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)
            if audit and getattr(self, "_audited", None):
                audit.append("request", path=self._audited, status=code, client=self.client_address[0],
                             **self._fields)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n == 0:
                return {}
            body = json.loads(self.rfile.read(n))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            self._fields = {k: str(body[k])[:200] for k in LOGGED_FIELDS if k in body}
            return body

        def _refuse(self, code: int, reason: str, say: str) -> bool:
            if audit:
                audit.append("refused", path=self.path, status=code, reason=reason, client=self.client_address[0])
            self._audited = None
            self._send(code, {"error": reason, "say": say})
            return False

        def _allowed(self) -> bool:
            """Host, Origin and token checks, in that order. Sends the refusal itself."""
            if enforce_host and _hostname(self.headers.get("Host") or "") not in LOOPBACK:
                return self._refuse(403, "unexpected Host header", "Request refused.")
            origin = self.headers.get("Origin")
            if origin and origin.rstrip("/") not in origins:
                return self._refuse(403, "origin not allowed", "Request refused.")
            if token:
                supplied = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
                if not hmac.compare_digest(supplied.encode(), token.encode()):
                    return self._refuse(401, "unauthorized", "Not authorized.")
            return True

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.end_headers()

        def do_GET(self):
            self._audited, self._fields = None, {}
            if not self._allowed():
                return
            path = self.path.rstrip("/")
            if path in ("/status", ""):
                self._send(200, assistant.status())
            elif path == "/audit":
                v = audit.verify() if audit else None
                self._send(200, {"ok": v.ok, "entries": v.entries, "head": v.head, "error": v.error}
                           if v else {"ok": None, "error": "audit log is not enabled"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.rstrip("/")
            self._audited, self._fields = None, {}
            if not self._allowed():
                return
            self._audited = path
            try:
                body = self._body()
                if path in ("/handoff", "/stage", "/resolve"):
                    if "text" in body and not body.get("instrument"):
                        body = parse_freeform(str(body["text"]))
                    r = catalog.resolve(
                        body.get("instrument", ""), body.get("suture"), body.get("size"), body.get("orientation")
                    )
                    info = {"slot": r.slot.id if r.slot else None, "instrument": r.instrument.id, "orientation": r.orientation,
                            "say": r.describe()}
                    if path == "/resolve":
                        return self._send(200, info)
                    job = assistant.submit(path.lstrip("/"), r)
                    return self._send(202, {"job": job, **info})
                if path == "/release":
                    return self._send(200, assistant.release())
                if path in ("/return", "/home"):
                    return self._send(202, {"job": assistant.submit(path.lstrip("/"))})
                if path == "/stop":
                    assistant.stop()
                    return self._send(200, {"status": "stopped"})
                if path == "/reset":
                    assistant.reset()
                    return self._send(200, {"status": assistant.phase.value})
                self._send(404, {"error": "not found"})
            except ResolveError as e:
                self._send(400, {"error": str(e), "say": str(e).capitalize()})
            except Busy as e:
                self._send(409, {"error": str(e), "say": "One moment, I'm still moving."})
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"bad request: {e}"})

    return Handler


def serve(assistant: ScrubAssistant, catalog: Catalog, host: str, port: int, token: str | None = None,
          allowed_origins=(), audit: AuditLog | None = None) -> ThreadingHTTPServer:
    handler = make_handler(assistant, catalog, token, allowed_origins, audit, enforce_host=host in LOOPBACK)
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd
