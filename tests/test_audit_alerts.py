import json

import pandas as pd

from src.common import load_data
from src.stream.audit import AuditLog, verify_file
from src.stream.pipeline import StreamPipeline


def test_audit_chain_detects_tampering(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl", key=b"k")
    for i in range(5):
        log.append("X", {"i": i}, "sha")
    assert verify_file(tmp_path / "a.jsonl", b"k")["ok"]
    assert not verify_file(tmp_path / "a.jsonl", b"wrong-key")["ok"]
    lines = (tmp_path / "a.jsonl").read_text().splitlines()
    rec = json.loads(lines[2]); rec["payload"]["i"] = 999
    lines[2] = json.dumps(rec, sort_keys=True)
    (tmp_path / "a.jsonl").write_text("\n".join(lines) + "\n")
    res = verify_file(tmp_path / "a.jsonl", b"k")
    assert not res["ok"] and res["first_bad_seq"] == 3


def test_audit_chain_detects_deleted_line(tmp_path):
    log = AuditLog(tmp_path / "b.jsonl")
    for i in range(4):
        log.append("X", {"i": i})
    lines = (tmp_path / "b.jsonl").read_text().splitlines()
    (tmp_path / "b.jsonl").write_text("\n".join(lines[:1] + lines[2:]) + "\n")
    assert not verify_file(tmp_path / "b.jsonl")["ok"]


def test_replay_issues_alerts_with_reasons_and_dedup(tmp_path):
    events, cells = load_data()
    p = StreamPipeline(cells, audit_path=tmp_path / "r.jsonl")
    res = p.replay(events, "2025-06-01", n_days=3, provisional_every_h=12)
    assert len(res) == 3 and all(r["kind"] == "OFFICIAL" for r in res)
    a = res[0]["alerts"]
    assert a and all(x["reasons"] and x["human_review_required"] for x in a)
    again = p.issue("2025-06-01", "OFFICIAL", log=False)["alerts"]
    assert all(x["status"] == "UNCHANGED" for x in again)           # no alert spam on re-score
    assert p.audit.verify()["ok"] and p.audit.verify()["entries"] == 3
