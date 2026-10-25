"""Hands-free listening: which overheard sentences are meant for the robot.

The mic is always on, so most of what it hears is ordinary operating-room talk. A sentence is acted on only if
  - it starts with the wake word ("Scrub, Potts scissors"), or
  - it comes within a few seconds of the robot speaking (a follow-up: "approve", "the left one"), or
  - it is "got it" / "release" while the arm is holding an instrument out, or
  - it is a short "stop" / "freeze" / "halt": stopping never needs the wake word (freezing is always safe).
Everything else is dropped here and never logged, stored or shown.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Callable

WAKE = re.compile(r"^\W*(?:(?:hey|ok|okay)\W+)?(?:scrub|scrubs|scrap)\b\W*", re.I)
STOP = re.compile(r"\b(stop|freeze|halt)\b", re.I)
RELEASE = re.compile(r"\b(got it|i('ve| have) got it|release|let go|take it|thanks?)\b", re.I)
PRESENTING = ("presenting", "staged")


class WakeGate:
    def __init__(self, follow_up_s: float = 8.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.follow_up_s, self.clock = follow_up_s, clock
        self._until = 0.0
        self._lock = threading.Lock()

    def open(self) -> None:
        """The robot just spoke or acted: listen without the wake word for a few seconds."""
        with self._lock:
            self._until = self.clock() + self.follow_up_s

    @property
    def awake(self) -> bool:
        return self.clock() < self._until

    def check(self, text: str, phase: str | None = None) -> str | None:
        """The request to act on (wake word removed; "" means only the wake word was said), or None to ignore it."""
        text = (text or "").strip()
        if not text:
            return None
        m = WAKE.match(text)
        if m:
            return text[m.end():].strip(" ,.!?")
        if self.awake:
            return text
        if phase in PRESENTING and RELEASE.search(text):
            return text
        if STOP.search(text) and len(text.split()) <= 3:
            return text
        return None
