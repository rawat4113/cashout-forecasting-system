"""
Tamper-evident audit log (hash chain, optional HMAC).

Every forecast, alert and analyst action is appended as one JSON line:
    hash_n = SHA256( hash_{n-1} || canonical_json(entry_n) )      (HMAC-SHA256 if a key is configured)
Editing, deleting or re-ordering any past line breaks every later hash, which `verify()` detects.

This is software-level integrity. On IBM Z the natural hardening is to keep the HMAC/signing key in the
Crypto Express hardware security module (via ICSF / PKCS#11) and to store the log on encrypted volumes;
that integration is a deployment step and is NOT implemented here.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import time
from pathlib import Path

GENESIS = "0" * 64


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


class AuditLog:
    def __init__(self, path, key: bytes | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.get("AUDIT_HMAC_KEY")
        self.key = key if key is not None else (env.encode() if env else None)
        self._lock = threading.Lock()
        self.seq, self.prev = 0, GENESIS
        if self.path.exists() and self.path.stat().st_size:
            last = self.path.read_text().strip().splitlines()[-1]
            rec = json.loads(last)
            self.seq, self.prev = rec["seq"], rec["hash"]

    def _digest(self, prev: str, body: dict) -> str:
        msg = prev.encode() + _canon(body)
        if self.key:
            return hmac.new(self.key, msg, hashlib.sha256).hexdigest()
        return hashlib.sha256(msg).hexdigest()

    def append(self, kind: str, payload: dict, model_sha256: str = "") -> dict:
        with self._lock:
            self.seq += 1
            body = dict(seq=self.seq, ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), kind=kind,
                        model_sha256=model_sha256, payload=payload)
            rec = dict(body, prev=self.prev, hash=self._digest(self.prev, body))
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            self.prev = rec["hash"]
            return rec

    def verify(self) -> dict:
        return verify_file(self.path, self.key)


def verify_file(path, key: bytes | None = None) -> dict:
    path = Path(path)
    if not path.exists():
        return dict(ok=True, entries=0, first_bad_seq=None, detail="no log yet")
    prev, n = GENESIS, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        n += 1
        body = {k: v for k, v in rec.items() if k not in ("prev", "hash")}
        msg = prev.encode() + _canon(body)
        want = hmac.new(key, msg, hashlib.sha256).hexdigest() if key else hashlib.sha256(msg).hexdigest()
        if rec["prev"] != prev or rec["hash"] != want or rec["seq"] != n:
            return dict(ok=False, entries=n, first_bad_seq=rec.get("seq"), detail="chain broken")
        prev = rec["hash"]
    return dict(ok=True, entries=n, first_bad_seq=None, detail="chain intact")
