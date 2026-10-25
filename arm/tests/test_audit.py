import json

import pytest

from scrub_arm.audit import GENESIS, AuditLog, verify


def _log(tmp_path, n=4, key=None):
    log = AuditLog(tmp_path / "audit.jsonl", key=key)
    for i in range(n):
        log.append("request", path="/handoff", instrument=f"tool{i}")
    return log


def _lines(path):
    return path.read_text().splitlines()


def test_empty_and_missing_logs_are_valid(tmp_path):
    v = verify(tmp_path / "nope.jsonl")
    assert v.ok and v.entries == 0 and v.head == GENESIS


def test_chain_verifies_and_head_advances(tmp_path):
    log = _log(tmp_path)
    v = log.verify()
    assert v.ok and v.entries == 4 and (4, v.head) == log.head()


def test_editing_an_entry_is_detected_with_its_line(tmp_path):
    log = _log(tmp_path)
    lines = _lines(log.path)
    e = json.loads(lines[1])
    e["data"]["instrument"] = "something else"
    lines[1] = json.dumps(e, sort_keys=True)
    log.path.write_text("\n".join(lines) + "\n")
    v = verify(log.path)
    assert not v.ok and "line 2" in v.error and v.entries == 1


def test_deleting_or_reordering_entries_is_detected(tmp_path):
    log = _log(tmp_path)
    lines = _lines(log.path)
    log.path.write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    assert not verify(log.path).ok
    log.path.write_text("\n".join([lines[0], lines[2], lines[1], lines[3]]) + "\n")
    assert not verify(log.path).ok


def test_reopening_continues_the_chain(tmp_path):
    log = _log(tmp_path, 2)
    again = AuditLog(log.path)
    again.append("request", path="/home")
    assert verify(log.path).entries == 3


def test_refuses_to_append_to_a_broken_log(tmp_path):
    log = _log(tmp_path)
    log.path.write_text(log.path.read_text().replace("tool1", "tool9"))
    with pytest.raises(ValueError, match="not intact"):
        AuditLog(log.path)


def test_key_stops_a_recomputed_chain(tmp_path):
    log = _log(tmp_path, 3, key="secret")
    assert verify(log.path, "secret").ok
    assert not verify(log.path, "wrong").ok
    # An attacker without the key rewrites an entry and recomputes plain-SHA chain hashes: still rejected.
    forged = AuditLog(tmp_path / "forged.jsonl")
    for i in range(3):
        forged.append("request", path="/handoff", instrument=f"tool{i}")
    assert verify(forged.path).ok and not verify(forged.path, "secret").ok


def test_truncating_the_tail_is_valid_but_changes_the_head(tmp_path):
    log = _log(tmp_path)
    seq, head = log.head()
    log.path.write_text("\n".join(_lines(log.path)[:-1]) + "\n")
    v = verify(log.path)
    assert v.ok and v.entries == seq - 1 and v.head != head     # why the head must be published elsewhere
