"""Map an intent ({instrument, suture, size, orientation}) to a physical tray slot.

The voice pipeline's LLM produces loosely-worded fields ("castro", "six-oh",
"backwards"). This module normalizes them against tray.yaml and picks the slot.
Anything ambiguous or unavailable raises `ResolveError` with a message that can
be read back to the surgeon, rather than guessing.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from .config import Instrument, Slot, TrayConfig

# Categories whose members are interchangeable (perception config/instruments.yaml `group`): a generic request
# ("scalpel") needs no further detail in camera mode. All scalpel blades count the same (user decision).
INTERCHANGEABLE_CATEGORIES = {"scalpel"}

_WORD_NUMBERS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}

ORIENTATIONS = {
    "forehand": "forehand", "fore": "forehand", "forward": "forehand", "forwards": "forehand",
    "front": "forehand", "normal": "forehand",
    "backhand": "backhand", "back": "backhand", "backward": "backhand", "backwards": "backhand",
    "reverse": "backhand", "reversed": "backhand",
}


class ResolveError(ValueError):
    pass


@dataclass
class Resolved:
    slot: Slot | None         # None in ik mode: the camera decides where the instrument is
    instrument: Instrument
    orientation: str | None   # None for instruments where orientation doesn't apply

    def describe(self) -> str:
        parts = [self.instrument.name]
        if self.slot and self.slot.size:
            parts.append(self.slot.size)
        if self.slot and self.slot.suture:
            parts.append(self.slot.suture.capitalize())
        if self.orientation:
            parts.append(self.orientation)
        return ", ".join(parts)


def _norm(text: str) -> str:
    text = text.lower().replace("_", " ").replace("’", "'")
    text = re.sub(r"[^a-z0-9#.\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _digits(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


def normalize_size(raw: str | None) -> str | None:
    """'6-0', '6/0', '6 0', 'six-oh', 'six 0' -> '6-0';  '1.5 mm', '1.5' -> '1.5mm'."""
    if raw is None or str(raw).strip() == "":
        return None
    s = _norm(str(raw)).replace("-", " ").replace("/", " ")
    s = " ".join(_WORD_NUMBERS.get(tok, tok) for tok in s.split())
    if "mm" in s or "millimet" in s or "." in s:
        m = re.search(r"\d+(?:\.\d+)?", s)
        if m:
            return f"{float(m.group()):g}mm"
    nums = re.findall(r"\d+", s)
    if len(nums) == 2 and nums[1] == "0":          # "6 0", "size 10 0"
        return f"{int(nums[0])}-0"
    if len(nums) == 1 and re.fullmatch(r"[2-9]0", nums[0]):  # "60" (STT dropped the dash)
        return f"{nums[0][0]}-0"
    return s.replace(" ", "")


def normalize_orientation(raw: str | None) -> str | None:
    if raw is None or str(raw).strip() == "":
        return None
    key = _norm(str(raw)).replace(" ", "").replace("-", "")
    if key in ORIENTATIONS:
        return ORIENTATIONS[key]
    for word in _norm(str(raw)).split():
        if word in ORIENTATIONS:
            return ORIENTATIONS[word]
    raise ResolveError(f"unknown orientation {raw!r}; say forehand or backhand")


def parse_freeform(text: str) -> dict:
    """Split 'needle holder, 6-0 prolene, backhand' into intent fields.

    Fallback for when the LLM intent parser is down, and for typing commands in
    the CLI. First comma chunk is the instrument; later chunks are classified.
    """
    chunks = [c.strip() for c in text.split(",") if c.strip()]
    if not chunks:
        return {"instrument": ""}
    out: dict = {"instrument": chunks[0]}
    for chunk in chunks[1:]:
        try:
            if normalize_orientation(chunk):
                out["orientation"] = chunk
                continue
        except ResolveError:
            pass
        sized, rest = [], []
        for word in chunk.split():
            (sized if re.search(r"\d", word) or word.lower() in _WORD_NUMBERS or word.lower() == "mm" else rest).append(word)
        if sized:
            out["size"] = " ".join(sized)
        if rest:
            out["suture"] = " ".join(rest)
    return out


class Catalog:
    def __init__(self, tray: TrayConfig, default_orientation: str = "forehand", require_slot: bool = True) -> None:
        self.tray = tray
        self.default_orientation = default_orientation
        self.require_slot = require_slot   # False (ik mode): any known instrument resolves, slot=None
        # phrase -> instrument id, or "category:<name>" for generic requests
        self._phrases: dict[str, str] = {}
        for inst in tray.instruments.values():
            for phrase in [inst.id, inst.name, *inst.aliases]:
                self._phrases.setdefault(_norm(phrase), inst.id)
            cat = _norm(inst.category)
            self._phrases.setdefault(cat, f"category:{inst.category}")
            self._phrases.setdefault(cat + "s", f"category:{inst.category}")
        self._sutures: dict[str, str] = {}
        for canon, aliases in tray.sutures.items():
            for phrase in [canon, *aliases]:
                self._sutures[_norm(phrase)] = canon

    def normalize_suture(self, raw: str | None) -> str | None:
        if raw is None or str(raw).strip() == "":
            return None
        s = _norm(str(raw))
        if s in self._sutures:
            return self._sutures[s]
        for phrase, canon in self._sutures.items():
            if phrase in s:
                return canon
        close = difflib.get_close_matches(s, list(self._sutures), n=1, cutoff=0.8)
        if close:
            return self._sutures[close[0]]
        raise ResolveError(f"unknown suture {raw!r}")

    def _match_instrument(self, raw: str) -> str:
        """Return an instrument id or 'category:<name>'."""
        s = _norm(raw)
        if s in self._phrases:
            return self._phrases[s]
        # Longest known phrase contained in the request ("give me the potts scissors please").
        contained = [p for p in self._phrases if re.search(rf"(?<![\w#]){re.escape(p)}(?![\w])", s)]
        if contained:
            best = max(contained, key=len)
            return self._phrases[best]
        # Fuzzy, but never across different numbers: "11 blade" must not become "10 blade".
        want = _digits(s)
        candidates = [p for p in self._phrases if _digits(p) == want]
        close = difflib.get_close_matches(s, candidates, n=2, cutoff=0.75)
        if close and (len(close) == 1 or self._phrases[close[0]] == self._phrases[close[1]]):
            return self._phrases[close[0]]
        raise ResolveError(f"I don't know the instrument {raw!r}")

    def resolve(self, instrument: str, suture: str | None = None, size: str | None = None,
                orientation: str | None = None) -> Resolved:
        if not instrument or not str(instrument).strip():
            raise ResolveError("no instrument specified")
        match = self._match_instrument(str(instrument))
        suture_n = self.normalize_suture(suture)
        size_n = normalize_size(size)
        orient_n = normalize_orientation(orientation)

        if not self.require_slot and match.startswith("category:"):
            cat = match.split(":", 1)[1]
            if cat in INTERCHANGEABLE_CATEGORIES:      # any member will do; the planner accepts the whole group
                match = next(i for i, inst in self.tray.instruments.items() if inst.category == cat)
            else:
                raise ResolveError(f"which {cat.replace('_', ' ')}? say its name")
        if not self.require_slot:
            inst = self.tray.instruments[match]
            orient = (orient_n or self.default_orientation) if inst.orientable else None
            return Resolved(slot=None, instrument=inst, orientation=orient)

        if match.startswith("category:"):
            cat = match.split(":", 1)[1]
            slots = [s for s in self.tray.slots if self.tray.instruments[s.instrument].category == cat]
            label = cat.replace("_", " ")
        else:
            slots = [s for s in self.tray.slots if s.instrument == match]
            label = self.tray.instruments[match].name
        if not slots:
            raise ResolveError(f"there is no {label} on the tray")

        if suture_n:
            slots_s = [s for s in slots if s.suture == suture_n]
            if not slots_s:
                have = sorted({f"{s.size or ''} {s.suture}".strip() for s in slots if s.suture})
                raise ResolveError(f"no {label} loaded with {suture_n}" + (f"; have {', '.join(have)}" if have else ""))
            slots = slots_s
        if size_n:
            slots_z = [s for s in slots if s.size == size_n]
            if not slots_z:
                have = sorted({s.size for s in slots if s.size})
                raise ResolveError(f"no {size_n} {label}" + (f"; have {', '.join(have)}" if have else ""))
            slots = slots_z

        slot = slots[0]
        inst = self.tray.instruments[slot.instrument]
        orient = (orient_n or self.default_orientation) if inst.orientable else None
        return Resolved(slot=slot, instrument=inst, orientation=orient)

    def slot(self, slot_id: str) -> Slot:
        for s in self.tray.slots:
            if s.id == slot_id:
                return s
        raise ResolveError(f"no slot {slot_id!r}")
