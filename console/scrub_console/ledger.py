"""Instrument ledger: where every physical instrument should be, and the end-of-case count check.

An instrument is on the TRAY, HELD by the arm, IN_FIELD (handed to the surgeon) or USED (returned to the
used zone). `reconcile` compares what the ledger expects to be visible (TRAY + USED) against what was
actually observed, and flags anything still outstanding (HELD / IN_FIELD). Instruments in the same group
(all scalpels, both micro scissors) are interchangeable, so seeing any one of them satisfies the others.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

TRAY, HELD, IN_FIELD, USED = "tray", "held", "in_field", "used"
STATUSES = (TRAY, HELD, IN_FIELD, USED)
# Which current states an instrument may move out of, per target state (a request names the instrument,
# not a particular physical copy).
_FROM = {
    HELD: (TRAY,),
    IN_FIELD: (HELD, TRAY),
    USED: (IN_FIELD, HELD),
    TRAY: (HELD, IN_FIELD, USED),
}
INSTRUMENTS_YAML = Path(__file__).resolve().parents[2] / "perception" / "config" / "instruments.yaml"


def load_groups(path: Path = INSTRUMENTS_YAML) -> dict[str, str]:
    """instrument id -> interchangeability group, from the perception config (empty if it is missing)."""
    try:
        d = yaml.safe_load(path.read_text())["instruments"]
    except (OSError, KeyError, TypeError):
        return {}
    return {k: v["group"] for k, v in d.items() if v.get("group")}


@dataclass
class Item:
    uid: str
    instrument: str
    status: str = TRAY


@dataclass
class Reconciliation:
    ok: bool
    missing: list[str] = field(default_factory=list)       # expected to be visible, not seen
    unexpected: list[str] = field(default_factory=list)    # seen, not expected
    outstanding: list[str] = field(default_factory=list)   # still held by the arm or in the field
    checked_visually: bool = False

    def to_dict(self) -> dict:
        return {"ok": self.ok, "missing": self.missing, "unexpected": self.unexpected,
                "outstanding": self.outstanding, "checked_visually": self.checked_visually}


class UnknownInstrument(KeyError):
    pass


class Ledger:
    def __init__(self, instruments: Iterable[str], names: dict[str, str] | None = None,
                 groups: dict[str, str] | None = None) -> None:
        self.names, self.groups = names or {}, groups or {}
        seen: Counter = Counter()
        self.items: list[Item] = []
        for iid in instruments:
            seen[iid] += 1
            self.items.append(Item(f"{iid}#{seen[iid]}", iid))

    @classmethod
    def from_tray(cls, tray, groups: dict[str, str] | None = None, in_set: bool = False) -> "Ledger":
        """One physical instrument per tray slot, or with `in_set` one per instrument in the set the real arm picks
        from (tray.yaml `in_set`; falls back to the slots when the set is empty)."""
        names = {i: v.name for i, v in tray.instruments.items()}
        ids = list(tray.in_set) if in_set and tray.in_set else [s.instrument for s in tray.slots]
        return cls(ids, names, load_groups() if groups is None else groups)

    def name(self, instrument: str) -> str:
        return self.names.get(instrument, instrument.replace("_", " "))

    def _key(self, instrument: str) -> str:
        return self.groups.get(instrument, instrument)

    def move(self, instrument: str, to: str) -> Item | None:
        """Move one copy of `instrument` into state `to`. Returns it, or None if no copy was in a state it may leave."""
        if not any(i.instrument == instrument for i in self.items):
            raise UnknownInstrument(instrument)
        for source in _FROM[to]:
            for item in self.items:
                if item.instrument == instrument and item.status == source:
                    item.status = to
                    return item
        return None

    def counts(self) -> dict[str, int]:
        c = Counter(i.status for i in self.items)
        return {s: c.get(s, 0) for s in STATUSES}

    def snapshot(self) -> list[dict]:
        return [{"uid": i.uid, "instrument": i.instrument, "name": self.name(i.instrument), "status": i.status}
                for i in self.items]

    def reconcile(self, observed: Iterable[str] | None = None) -> Reconciliation:
        """observed: instrument ids seen on the tray and in the used zone (None = no visual count available)."""
        outstanding = [self.name(i.instrument) for i in self.items if i.status in (HELD, IN_FIELD)]
        if observed is None:
            return Reconciliation(not outstanding, outstanding=outstanding)
        expected = Counter(self._key(i.instrument) for i in self.items if i.status in (TRAY, USED))
        seen = Counter(self._key(o) for o in observed)
        label = {self._key(i.instrument): self.name(i.instrument) for i in self.items}
        missing = [label.get(k, k) for k in (expected - seen).elements()]
        unexpected = [label.get(k, k.replace("_", " ")) for k in (seen - expected).elements()]
        return Reconciliation(not (missing or unexpected or outstanding), missing, unexpected, outstanding, True)

    def say_for(self, r: Reconciliation) -> str:
        """The sentence read aloud after a count check."""
        n = len(self.items)
        if r.ok:
            return f"Count correct. All {n} instruments are accounted for."
        parts = []
        if r.missing:
            parts.append("Missing: " + ", ".join(r.missing))
        if r.outstanding:
            parts.append("Still out: " + ", ".join(r.outstanding))
        if r.unexpected:
            parts.append("Unexpected on the table: " + ", ".join(r.unexpected))
        return "Count is off. " + ". ".join(parts) + "."
