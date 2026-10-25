"""Tamper-evident audit log: JSON lines, each entry carrying a hash of the entry before it.

Editing, deleting or reordering any entry breaks the chain from that point on, and `verify` reports
where. Two limits to know about:
  - Without a key anyone who can write the file can recompute the whole chain. Set SCRUB_AUDIT_KEY (or
    pass `key`) and entries are HMAC-SHA256 tags, which cannot be forged without it.
  - Cutting entries off the END leaves a valid chain. Publish `head()` somewhere else (the console does
    this) and compare it against `verify(...).head`.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

GENESIS = "0" * 64


def _digest(entry: dict, key: bytes | None) -> str:
    body = json.dumps({k: entry[k] for k in ("seq", "t", "kind", "data", "prev")}, sort_keys=True,
                      separators=(",", ":"), default=str).encode()
    if key:
        return hmac.new(key, body, hashlib.sha256).hexdigest()
    return hashlib.sha256(body).hexdigest()


@dataclass
class Verdict:
    ok: bool
    entries: int
    head: str
    error: str | None = None


def _key(key: bytes | str | None) -> bytes | None:
    key = os.environ.get("SCRUB_AUDIT_KEY") if key is None else key
    return key.encode() if isinstance(key, str) and key else (key or None)


def verify(path: Path | str, key: bytes | str | None = None) -> Verdict:
    """Walk the whole file and check every link. A missing file is an empty, valid log."""
    path, k = Path(path), _key(key)
    prev, n = GENESIS, 0
    if not path.exists():
        return Verdict(True, 0, prev)
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        try:
            e = json.loads(line)
            if e["seq"] != n + 1:
                return Verdict(False, n, prev, f"line {lineno}: expected entry {n + 1}, found {e['seq']}")
            if e["prev"] != prev:
                return Verdict(False, n, prev, f"line {lineno}: does not follow the entry before it")
            if not hmac.compare_digest(e["hash"], _digest(e, k)):
                return Verdict(False, n, prev, f"line {lineno}: contents were changed")
        except (ValueError, KeyError, TypeError):
            return Verdict(False, n, prev, f"line {lineno}: unreadable entry")
        prev, n = e["hash"], n + 1
    return Verdict(True, n, prev)


class AuditLog:
    def __init__(self, path: Path | str, key: bytes | str | None = None) -> None:
        self.path, self._key = Path(path), _key(key)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        v = verify(self.path, self._key)
        if not v.ok:
            raise ValueError(f"audit log {self.path} is not intact: {v.error}")
        self._seq, self._head = v.entries, v.head

    def append(self, kind: str, /, **data) -> dict:
        with self._lock:
            entry = {"seq": self._seq + 1, "t": round(time.time(), 3), "kind": kind, "data": data, "prev": self._head}
            entry["hash"] = _digest(entry, self._key)
            with open(self.path, "a") as f:
                f.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._seq, self._head = entry["seq"], entry["hash"]
            return entry

    def head(self) -> tuple[int, str]:
        with self._lock:
            return self._seq, self._head

    def verify(self) -> Verdict:
        with self._lock:
            return verify(self.path, self._key)
