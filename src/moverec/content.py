"""Content side: move-profile vectors + cosine nearest neighbours.

Built for the empty cart: when nothing is in the cart yet, the only signal is
the move itself. We find the k training bookings whose move profile is most
similar and recommend what they bought.

Profile vector (all fitted on training data only):
    home_size      ordinal 0..4 (studio .. 4br_plus), so 2br is near 3br
    labor_only     0/1
    distance       log miles
    lead_time      log(1 + days)
    origin_access  one-hot (3)
    dest_access    one-hot (3)
    move_season    sin/cos of move month (December sits next to January)

Every column is standardised. Each feature GROUP is then scaled to the same
total variance (a 3-column one-hot doesn't outvote a single column), times an
optional group weight.

This is a smoother, richer version of profile popularity: profile popularity
only matches on move type x home size x distance band, while the kNN also uses
lead time, access and season, which the generator does use.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOME_SIZE_ORDER = ["studio", "1br", "2br", "3br", "4br_plus"]   # public schema order
ACCESS_TYPES = ["ground", "stairs", "elevator"]


def _raw_groups(bookings: pd.DataFrame) -> dict[str, np.ndarray]:
    month = pd.to_datetime(bookings["move_date"]).dt.month.to_numpy()
    angle = 2 * np.pi * (month - 1) / 12
    return {
        "home_size": bookings["home_size"].map({h: i for i, h in enumerate(HOME_SIZE_ORDER)})
                     .fillna(2).to_numpy(float)[:, None],
        "labor_only": (bookings["move_type"] == "labor_only").to_numpy(float)[:, None],
        "distance": np.log(bookings["distance_miles"].clip(lower=1)).to_numpy(float)[:, None],
        "lead_time": np.log1p(bookings["lead_time_days"].clip(lower=0)).to_numpy(float)[:, None],
        "origin_access": np.column_stack([(bookings["origin_access"] == a).to_numpy(float) for a in ACCESS_TYPES]),
        "dest_access": np.column_stack([(bookings["dest_access"] == a).to_numpy(float) for a in ACCESS_TYPES]),
        "move_season": np.column_stack([np.sin(angle), np.cos(angle)]),
    }


GROUPS = list(_raw_groups(pd.DataFrame({
    "move_date": ["2025-01-01"], "home_size": ["2br"], "move_type": ["truck"], "distance_miles": [10],
    "lead_time_days": [7], "origin_access": ["ground"], "dest_access": ["ground"]})))


class ProfileEncoder:
    """Standardise + group-balance the profile features (fit on training data)."""

    def __init__(self, group_weights: dict[str, float] | None = None):
        self.group_weights = group_weights or {}
        unknown = set(self.group_weights) - set(GROUPS)
        if unknown:
            raise ValueError(f"unknown feature groups: {sorted(unknown)}")

    def fit(self, bookings: pd.DataFrame) -> "ProfileEncoder":
        raw = _raw_groups(bookings)
        self.mean_ = {g: v.mean(axis=0) for g, v in raw.items()}
        self.std_ = {g: np.where(v.std(axis=0) > 0, v.std(axis=0), 1.0) for g, v in raw.items()}
        return self

    def transform(self, bookings: pd.DataFrame) -> np.ndarray:
        raw = _raw_groups(bookings)
        blocks = []
        for g in GROUPS:
            z = (raw[g] - self.mean_[g]) / self.std_[g]
            z /= np.sqrt(z.shape[1])                        # equal variance per group
            blocks.append(self.group_weights.get(g, 1.0) * z)
        return np.hstack(blocks)


class ContentKNN:
    """Score services by what the k most similar past moves bought. Ignores the cart."""
    name = "content_knn"

    def __init__(self, k: int = 100, group_weights: dict[str, float] | None = None,
                 batch_size: int = 1024):
        self.k, self.group_weights, self.batch_size = k, group_weights, batch_size

    @staticmethod
    def _unit(V: np.ndarray) -> np.ndarray:
        return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

    def fit(self, bookings: pd.DataFrame, X: np.ndarray) -> "ContentKNN":
        self.encoder_ = ProfileEncoder(self.group_weights).fit(bookings)
        self.train_vecs_ = self._unit(self.encoder_.transform(bookings))
        self.X_ = X.astype(float)
        self.popularity_ = X.mean(axis=0)
        return self

    def neighbours(self, bookings: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(indices, cosine similarities) of the k nearest training bookings, best first."""
        Q = self._unit(self.encoder_.transform(bookings))
        k = min(self.k, len(self.train_vecs_))
        idx_out, sim_out = [], []
        for start in range(0, len(Q), self.batch_size):
            sims = Q[start:start + self.batch_size] @ self.train_vecs_.T
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            s = np.take_along_axis(sims, idx, axis=1)
            order = np.argsort(-s, axis=1, kind="stable")
            idx_out.append(np.take_along_axis(idx, order, axis=1))
            sim_out.append(np.take_along_axis(s, order, axis=1))
        return np.vstack(idx_out), np.vstack(sim_out)

    def score(self, bookings: pd.DataFrame, cart: np.ndarray) -> np.ndarray:
        idx, sim = self.neighbours(bookings)
        w = np.clip(sim, 0.0, None)                         # ignore dissimilar moves
        votes = np.einsum("nk,nks->ns", w, self.X_[idx])
        total = w.sum(axis=1, keepdims=True)
        scores = np.where(total > 0, votes / np.maximum(total, 1e-12), self.popularity_)
        return scores + 1e-6 * self.popularity_
