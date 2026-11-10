"""Baseline recommenders.

All models share one interface:
    fit(bookings, X)           X = bool (n_bookings, n_services) training baskets
    score(bookings, cart)      -> float (n, n_services), higher = recommend first

"Segment popularity" here uses OBSERVED profile segments (move type, home
size, distance band). The generator's hidden segments are never available.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROFILE_LEVELS = [
    ("move_type",),
    ("move_type", "home_size"),
    ("move_type", "home_size", "distance_band"),
]


class GlobalPopularity:
    """Rank services by training attach rate. Ignores profile and cart."""
    name = "global_popularity"

    def fit(self, bookings: pd.DataFrame, X: np.ndarray) -> "GlobalPopularity":
        self.rate_ = X.mean(axis=0)
        return self

    def score(self, bookings: pd.DataFrame, cart: np.ndarray) -> np.ndarray:
        return np.tile(self.rate_, (len(cart), 1))


class ProfilePopularity:
    """Attach rate within the booking's observed profile segment.

    Rates are shrunk level by level toward the parent level:
        rate(level) = (purchases + m * rate(parent)) / (bookings + m)
    so small segments fall back gracefully to coarser ones, and finally to the
    global rate. Ignores the cart.
    """
    name = "profile_popularity"

    def __init__(self, smoothing: float = 20.0, levels=PROFILE_LEVELS):
        self.smoothing = smoothing
        self.levels = levels

    def fit(self, bookings: pd.DataFrame, X: np.ndarray) -> "ProfilePopularity":
        self.global_ = X.mean(axis=0)
        self.tables_ = []
        for cols in self.levels:
            codes, keys = pd.factorize(pd.Series(list(zip(*(bookings[c] for c in cols)))))
            self.tables_.append({
                key: (X[codes == i].sum(axis=0).astype(float), int((codes == i).sum()))
                for i, key in enumerate(keys)
            })
        return self

    def rate_for(self, profile: dict) -> np.ndarray:
        rate = self.global_
        for cols, table in zip(self.levels, self.tables_):
            key = tuple(profile[c] for c in cols)
            if key in table:
                s, n = table[key]
                rate = (s + self.smoothing * rate) / (n + self.smoothing)
        return rate

    def score(self, bookings: pd.DataFrame, cart: np.ndarray) -> np.ndarray:
        full = list(self.levels[-1])
        keys = bookings[full].apply(tuple, axis=1)
        cache = {k: self.rate_for(dict(zip(full, k))) for k in keys.unique()}
        return np.vstack([cache[k] for k in keys])


class CoOccurrence:
    """Score j by the mean of P(j | i) over services i in the cart.

    Empty carts fall back to global popularity (a tiny popularity term also
    breaks ties). Ignores the profile.
    """
    name = "co_occurrence"

    def fit(self, bookings: pd.DataFrame, X: np.ndarray) -> "CoOccurrence":
        Xf = X.astype(float)
        counts = Xf.sum(axis=0)
        self.cond_ = (Xf.T @ Xf) / np.maximum(counts, 1.0)[:, None]  # row i: P(j | i)
        np.fill_diagonal(self.cond_, 0.0)
        self.rate_ = X.mean(axis=0)
        return self

    def score(self, bookings: pd.DataFrame, cart: np.ndarray) -> np.ndarray:
        c = cart.astype(float)
        mean_cond = (c @ self.cond_) / np.maximum(c.sum(axis=1, keepdims=True), 1.0)
        return mean_cond + 1e-6 * self.rate_


class CoOccurrenceLift:
    """Cart signal with popularity divided out: mean over cart items i of log lift(i -> j).

        lift(i -> j) = P(j | i) / P(j),   P(j | i) shrunk toward P(j) by `prior` pseudo-bookings

    so a rarely seen item i gives lift ~ 1 (log 0) instead of a noisy estimate.
    Positive = "people with this cart buy j MORE than usual". Alone it is a poor
    ranker (it ignores how common j is). It is built to be added to a
    popularity-type score in the hybrid. Empty carts score 0 everywhere.
    """
    name = "cooc_lift"

    def __init__(self, prior: float = 10.0):
        self.prior = prior

    def fit(self, bookings: pd.DataFrame, X: np.ndarray) -> "CoOccurrenceLift":
        Xf = X.astype(float)
        n_i = Xf.sum(axis=0)
        p = np.maximum(X.mean(axis=0), 1.0 / (len(X) + 1))
        cond = (Xf.T @ Xf + self.prior * p[None, :]) / (n_i[:, None] + self.prior)
        self.log_lift_ = np.log(cond / p[None, :])
        np.fill_diagonal(self.log_lift_, 0.0)
        return self

    def score(self, bookings: pd.DataFrame, cart: np.ndarray) -> np.ndarray:
        c = cart.astype(float)
        return (c @ self.log_lift_) / np.maximum(c.sum(axis=1, keepdims=True), 1.0)
