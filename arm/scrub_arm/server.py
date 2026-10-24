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

Errors come back as {"error": ..., "say": ...}; `say` is short enough to feed straight to TTS.
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .catalog import Catalog, ResolveError, parse_freeform
from .handoff import Busy, ScrubAssistant

log = logging.getLogger(__name__)


def make_handler(assistant: ScrubAssistant, catalog: Catalog):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # route through logging instead of stderr
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n == 0:
                return {}
            body = json.loads(self.rfile.read(n))
            if not isinstance(body, dict):
                raise ValueError("body must be a JSON object")
            return body

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if self.path.rstrip("/") in ("/status", ""):
                self._send(200, assistant.status())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.rstrip("/")
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


def serve(assistant: ScrubAssistant, catalog: Catalog, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(assistant, catalog))
    httpd.daemon_threads = True
    return httpd
