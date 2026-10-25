"""Apply an approved remote config change to arm.yaml, safely.

The cloud already checks the allowlist and bounds; the robot checks them again (it's the last line
of defence for its own hardware), edits only that one value, keeps comments, and backs up the file.
scrub_arm reads arm.yaml at startup, so a change takes effect on the next `scrub_arm serve`.
"""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

# Must match TUNABLE in fleet/convex/lib.ts.
TUNABLE: dict[str, tuple[float, float]] = {
    "motion.speed_scale": (0.1, 0.6),
    "motion.approach_speed_scale": (0.05, 0.4),
    "motion.tracking_error_limit_deg": (10, 30),
    "motion.settle_timeout_s": (0.5, 4),
    "gripper.move_torque_nm": (0.5, 2.0),
    "gripper.grip_torque_nm": (0.3, 1.5),
    "gripper.holding_min_gap_deg": (1.0, 6.0),
    "pick.contact_deviation_deg": (0.8, 4.0),
}


class ConfigChangeError(ValueError):
    pass


def _value_span(text: str, key: str) -> tuple[int, int]:
    section, name = key.split(".", 1)
    sec = re.search(rf"^{re.escape(section)}:[ \t]*(#.*)?$", text, re.M)
    if not sec:
        raise ConfigChangeError(f"arm.yaml has no `{section}:` section")
    # The section runs until the next top-level key.
    end = re.search(r"^\S", text[sec.end() + 1:], re.M)
    body_end = sec.end() + 1 + end.start() if end else len(text)
    m = re.compile(rf"^([ \t]+{re.escape(name)}:[ \t]*)([^\s#]+)", re.M).search(text, sec.end(), body_end)
    if not m:
        raise ConfigChangeError(f"arm.yaml has no `{key}` (this arm's config may predate that setting)")
    return m.start(2), m.end(2)


def current_value(path: Path, key: str) -> float | None:
    text = path.read_text()
    a, b = _value_span(text, key)
    try:
        return float(text[a:b])
    except ValueError:
        return None


def apply_change(path: Path, key: str, to: float) -> str:
    if key not in TUNABLE:
        raise ConfigChangeError(f"{key} is not remotely tunable on this arm")
    lo, hi = TUNABLE[key]
    if not (lo <= to <= hi):
        raise ConfigChangeError(f"{key}={to} is outside {lo}..{hi}")
    text = path.read_text()
    a, b = _value_span(text, key)
    old = text[a:b]
    backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(path, backup)
    path.write_text(text[:a] + f"{to:g}" + text[b:])
    return f"{key}: {old} -> {to:g} (backup {backup.name}); restart `scrub_arm serve` to load it"


def excerpt(path: Path) -> dict[str, float | None]:
    """Current values of every tunable key, for incident context."""
    out = {}
    for k in TUNABLE:
        try:
            out[k] = current_value(path, k)
        except ConfigChangeError:
            out[k] = None
    return out
