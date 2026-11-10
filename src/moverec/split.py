"""Time-based train / validation / test split on booking time."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def parts(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "val": self.val, "test": self.test}


def time_split(bookings: pd.DataFrame, val_start: str, test_start: str) -> Split:
    """Everything before val_start trains; [val_start, test_start) validates; the rest tests."""
    v, t = pd.Timestamp(val_start), pd.Timestamp(test_start)
    if not v < t:
        raise ValueError("val_start must be before test_start")
    b = bookings.sort_values("booking_ts")
    ts = b["booking_ts"]
    return Split(
        train=b[ts < v].reset_index(drop=True),
        val=b[(ts >= v) & (ts < t)].reset_index(drop=True),
        test=b[ts >= t].reset_index(drop=True),
    )


def split_summary(split: Split) -> pd.DataFrame:
    """Size, date range and cold-start share of each part.

    A booking is cold start if its customer has no booking before the start of
    its part (for train: no earlier booking at all).
    """
    all_b = pd.concat(split.parts().values())
    rows = []
    for name, part in split.parts().items():
        if part.empty:
            continue
        start = part["booking_ts"].min()
        if name == "train":
            cold = ~part.sort_values("booking_ts").duplicated("customer_id")
        else:
            seen = set(all_b.loc[all_b["booking_ts"] < start, "customer_id"])
            cold = ~part["customer_id"].isin(seen)
        rows.append({
            "part": name,
            "bookings": len(part),
            "from": f"{start:%Y-%m-%d}",
            "to": f"{part['booking_ts'].max():%Y-%m-%d}",
            "cold_start_share": round(float(cold.mean()), 3),
        })
    return pd.DataFrame(rows)
