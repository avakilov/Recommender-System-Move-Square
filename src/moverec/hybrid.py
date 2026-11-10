"""Weighted hybrid of the profile, content and cart (collaborative) signals.

Each component's scores are turned into one comparable scale, then summed with
weights tuned on validation:

    hybrid(j) = sum_c  w_c * standardise_row( transform_c(score_c(j)) )

    component   what it knows            transform
    profile     move profile segment     log  (probabilities multiply -> logs add)
    content     similar past moves       log
    cooc_lift   items in the cart        none (already a mean log lift)
    als         items in the cart        none

"standardise_row" = z-score across the 45 services within each cart, so no
component wins just because its numbers are bigger. A row with no variation
(e.g. the cart components on an empty cart) becomes all zeros, meaning that
component abstains and the others decide.

Only the RATIOS of the weights matter for ranking.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

LOG_COMPONENTS = {"profile", "content"}


def row_standardize(S: np.ndarray) -> np.ndarray:
    mu = S.mean(axis=1, keepdims=True)
    sd = S.std(axis=1, keepdims=True)
    return np.where(sd > 1e-12, (S - mu) / np.where(sd > 1e-12, sd, 1.0), 0.0)


def component_scores(components: dict, bookings: pd.DataFrame, cart: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for name, model in components.items():
        s = model.score(bookings, cart)
        if name in LOG_COMPONENTS:
            s = np.log(np.clip(s, 1e-4, None))
        out[name] = row_standardize(s)
    return out


def combine(scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    if not any(weights.get(name, 0.0) for name in scores):
        raise ValueError("all weights are zero")
    return sum(weights.get(name, 0.0) * s for name, s in scores.items())


class FixedScores:
    """Wrap a precomputed score matrix so evaluate() can rank it (used for fast tuning)."""

    def __init__(self, scores: np.ndarray, name: str = "fixed"):
        self.scores, self.name = scores, name

    def score(self, bookings, cart) -> np.ndarray:
        return self.scores


class Hybrid:
    name = "hybrid"

    def __init__(self, components: dict, weights: dict[str, float]):
        unknown = set(weights) - set(components)
        if unknown:
            raise ValueError(f"weights for unknown components: {sorted(unknown)}")
        self.components, self.weights = components, weights

    def score(self, bookings: pd.DataFrame, cart: np.ndarray) -> np.ndarray:
        return combine(component_scores(self.components, bookings, cart), self.weights)


def weight_grid(names: list[str], values: list[float], fixed_zero: tuple[str, ...] = ()):
    """Every weight combination over `values`, skipping all-zero ones."""
    free = [n for n in names if n not in fixed_zero]
    for combo in itertools.product(values, repeat=len(free)):
        if any(combo):
            w = dict(zip(free, combo))
            w.update({n: 0.0 for n in fixed_zero})
            yield w


def tune_weights(scores: dict[str, np.ndarray], evaluate_fn, values: list[float],
                 fixed_zero: tuple[str, ...] = ()) -> tuple[dict, float, pd.DataFrame]:
    """Grid-search weights. evaluate_fn(score_matrix) -> metric (higher is better).

    Ties go to the first combination in grid order, which favours smaller weights.
    """
    rows = []
    for w in weight_grid(list(scores), values, fixed_zero):
        rows.append({**w, "metric": evaluate_fn(combine(scores, w))})
    table = pd.DataFrame(rows)
    best = table.loc[table["metric"].idxmax()]
    return {n: float(best[n]) for n in scores}, float(best["metric"]), table
