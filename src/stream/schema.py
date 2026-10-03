"""Event contract for the complaint stream (what a Kafka topic / MQ queue would carry)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

KNOWN_CATEGORIES = [
    # sorted (code-point order) on purpose: ties in "dominant fraud type" then resolve exactly as in
    # the batch path (src/predict.py sorts by count desc, category asc). Unknown labels map to "Other".
    "Digital Arrest Scam", "Investment / Trading Scam", "Job / Task Fraud", "KYC / OTP Fraud",
    "Other", "Sextortion", "UPI / Card Fraud",
]


class EventValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ComplaintEvent:
    """A traced complaint that has reached the portal. No personal data is carried: only
    a pseudonymous id, time, ATM-cluster id, fraud category and amount."""
    event_id: int
    complaint_time: datetime
    cell_id: int
    category: str
    amount: float

    @staticmethod
    def from_dict(d: dict) -> "ComplaintEvent":
        try:
            ts = pd.Timestamp(d["complaint_time"]).floor("us").to_pydatetime()
            ev = ComplaintEvent(
                event_id=int(d["event_id"]),
                complaint_time=ts,
                cell_id=int(d["cell_id"]),
                category=str(d.get("category", "Other")),
                amount=float(d.get("amount", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EventValidationError(f"malformed event: {exc}") from exc
        if ev.amount < 0 or ev.amount != ev.amount:       # negative or NaN
            raise EventValidationError("amount must be a non-negative number")
        return ev

    def to_dict(self) -> dict:
        return dict(event_id=self.event_id, complaint_time=self.complaint_time.isoformat(),
                    cell_id=self.cell_id, category=self.category, amount=self.amount)


def category_index(name: str) -> int:
    try:
        return KNOWN_CATEGORIES.index(name)
    except ValueError:
        return KNOWN_CATEGORIES.index("Other")
