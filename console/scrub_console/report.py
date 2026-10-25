"""End-of-case report and next-instrument prediction, run on Bitdeer AI Model Studio.

Bitdeer's API is OpenAI-compatible chat completions (https://api-inference.bitdeer.ai/v1), so this is a
plain HTTP call. Neither feature is on the safety path: the report only restates facts recorded by the
console, and a prediction only ever becomes a suggestion on the dashboard that a person must act on.
Each result says what produced it ("bitdeer:<model>" or "template") so nobody mistakes one for the other.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

log = logging.getLogger(__name__)


class BitdeerError(RuntimeError):
    pass


class BitdeerClient:
    def __init__(self, api_key: str, base_url: str = "https://api-inference.bitdeer.ai/v1",
                 model: str = "deepseek-ai/DeepSeek-V4-Pro", timeout_s: float = 60.0) -> None:
        self.api_key, self.base_url, self.model, self.timeout_s = api_key, base_url.rstrip("/"), model, timeout_s

    def chat(self, messages: list[dict], max_tokens: int = 800) -> str:
        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": max_tokens}).encode()
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers={
            "Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError, IndexError) as e:
            raise BitdeerError(f"bitdeer request failed: {e}") from e

    @property
    def label(self) -> str:
        return f"bitdeer:{self.model}"


def build_facts(events: list[dict], counts: dict, audit: dict) -> dict:
    """Everything the report may say, taken only from recorded console events."""
    jobs = [e["data"] for e in events if e["kind"] == "job"]
    handoffs = [j for j in jobs if j["kind"] in ("handoff", "stage")]
    counts_seen = [e["data"] for e in events if e["kind"] == "count"]
    return {
        "requests": sum(1 for e in events if e["kind"] == "utterance"),
        "instrument_jobs": len(handoffs),
        "completed": sum(1 for j in handoffs if j["status"] == "done"),
        "failed": [{"kind": j["kind"], "error": j.get("error")} for j in handoffs if j["status"] != "done"],
        "faults": [e["data"]["message"] for e in events if e["kind"] == "fault"],
        "proposals": [{"status": e["data"]["status"], "simulated": e["data"].get("simulated"),
                       "changes": e["data"].get("changes")} for e in events if e["kind"] == "proposal"][-3:],
        "last_count": ({k: counts_seen[-1][k] for k in ("ok", "missing", "outstanding", "unexpected", "source", "say")}
                       if counts_seen else None),
        "ledger": counts,
        "audit": {k: {"intact": v.get("ok"), "entries": v.get("entries")} for k, v in (audit or {}).items()},
    }


def template_report(facts: dict) -> str:
    lines = ["End-of-case report (template; no model was used)"]
    lines.append(f"- {facts['requests']} spoken requests; {facts['completed']} of {facts['instrument_jobs']} "
                 "instrument jobs completed.")
    for f in facts["failed"]:
        lines.append(f"- Failed {f['kind']}: {f['error']}")
    c = facts["last_count"]
    lines.append("- No count check was run." if not c else
                 f"- Last count ({c['source']}): {'correct' if c['ok'] else 'DISCREPANCY'}. {c['say']}")
    lines.append(f"- Faults: {len(facts['faults'])}" + (": " + "; ".join(facts["faults"]) if facts["faults"] else "."))
    for p in facts["proposals"]:
        lines.append(f"- Config proposal {p['status']}" + (" (rehearsal)" if p.get("simulated") else "") + ".")
    for name, a in facts["audit"].items():
        lines.append(f"- Audit log ({name}): {'intact' if a['intact'] else 'NOT INTACT'}, {a['entries']} entries.")
    return "\n".join(lines)


REPORT_PROMPT = (
    "You write the end-of-case note for the circulating nurse after a robotic scrub assistant session (a demo, "
    "not a medical device). Use ONLY the JSON facts below; do not invent anything. Be plain and brief (under "
    "150 words). Lead with whether the instrument count was correct and name any missing or outstanding "
    "instrument. Then faults and anything a person still needs to do.\n\nFacts:\n"
)


def summarize_case(facts: dict, client: BitdeerClient | None = None) -> dict:
    if client is not None:
        try:
            text = client.chat([{"role": "user", "content": REPORT_PROMPT + json.dumps(facts, indent=1, default=str)}])
            return {"text": text, "generated_by": client.label}
        except BitdeerError as e:
            log.warning("%s; using the template report", e)
    return {"text": template_report(facts), "generated_by": "template"}


def predict_next(history: list[str], remaining: dict[str, str], client: BitdeerClient | None) -> dict | None:
    """Suggest the instrument the surgeon will probably ask for next. `remaining` maps id -> display name
    for instruments still on the tray. Returns None without a model or if the model names anything not on the tray."""
    if client is None or not remaining:
        return None
    prompt = (
        "A surgeon has requested these instruments in this order: " + (", ".join(history) or "none yet") +
        ". Still on the tray (id: name): " + json.dumps(remaining) + ". Which ONE is most likely requested "
        'next? Answer with JSON only: {"instrument": "<id from the tray list>", "reason": "<one short sentence>"}.'
    )
    try:
        text = client.chat([{"role": "user", "content": prompt}], max_tokens=200)
        m = re.search(r"\{.*?\}", text, re.S)
        choice = json.loads(m.group()) if m else {}
    except (BitdeerError, ValueError):
        return None
    if choice.get("instrument") not in remaining:
        return None
    return {"instrument": choice["instrument"], "name": remaining[choice["instrument"]],
            "reason": str(choice.get("reason", ""))[:200], "generated_by": client.label}
