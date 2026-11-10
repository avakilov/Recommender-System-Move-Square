"""Offline evaluation on simulated partial carts.

Protocol (identical for every model):
  * For each held-out booking with at least one service, keep a random subset
    of its services as the visible cart (possibly empty) and hide the rest.
    At least one service is always hidden. The hidden services are the targets.
  * The model scores every service given the booking profile and the cart.
  * Services already in the cart, and services the catalog rules make
    impossible (a second truck, any truck on a labor-only move, a second
    storage unit), are removed before ranking. These rules apply to all models.
  * Metrics: hit rate@k, recall@k, MAP@k, averaged over carts.

Carts are drawn once from a fixed seed, so every model sees the same carts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Recommender(Protocol):
    name: str

    def fit(self, bookings: pd.DataFrame, X: np.ndarray) -> "Recommender": ...

    def score(self, bookings: pd.DataFrame, cart: np.ndarray) -> np.ndarray: ...


@dataclass
class EvalCarts:
    bookings: pd.DataFrame   # profile of each evaluated booking (row-aligned)
    cart: np.ndarray         # bool (n, S): visible services
    target: np.ndarray       # bool (n, S): hidden services to recover


def make_partial_carts(bookings: pd.DataFrame, X: np.ndarray, seed: int) -> EvalCarts:
    rng = np.random.default_rng(seed)
    keep_rows = np.flatnonzero(X.any(axis=1))
    cart = np.zeros((len(keep_rows), X.shape[1]), dtype=bool)
    for out_r, r in enumerate(keep_rows):
        services = np.flatnonzero(X[r])
        n_visible = rng.integers(0, len(services))  # 0 .. n-1, so >= 1 hidden
        cart[out_r, rng.choice(services, size=n_visible, replace=False)] = True
    target = X[keep_rows] & ~cart
    return EvalCarts(bookings.iloc[keep_rows].reset_index(drop=True), cart, target)


def feasibility_mask(bookings: pd.DataFrame, cart: np.ndarray, services: pd.DataFrame) -> np.ndarray:
    """True where a service may still be recommended under catalog rules."""
    group = services["group"].to_numpy()
    ok = ~cart.copy()
    truck, storage = group == "truck", group == "storage"
    labor_only = (bookings["move_type"] == "labor_only").to_numpy()
    has_truck = cart[:, truck].any(axis=1)
    has_storage = cart[:, storage].any(axis=1)
    ok[np.ix_(labor_only | has_truck, truck)] = False
    ok[np.ix_(has_storage, storage)] = False
    return ok


def top_k(scores: np.ndarray, allowed: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best allowed services per row. Ties -> catalog order."""
    s = np.where(allowed, scores, -np.inf)
    order = np.argsort(-s, axis=1, kind="stable")[:, :k]
    # a row may have fewer than k allowed services; mark those slots -1
    valid = np.take_along_axis(allowed, order, axis=1)
    return np.where(valid, order, -1)


def _row_metrics(ranked: np.ndarray, target: np.ndarray, ks: list[int], map_k: int) -> pd.DataFrame:
    n_target = target.sum(axis=1)
    safe = np.where(ranked >= 0, ranked, 0)
    rel = np.take_along_axis(target, safe, axis=1) & (ranked >= 0)
    out = {}
    for k in ks:
        hits = rel[:, :k].sum(axis=1)
        out[f"hit@{k}"] = (hits > 0).astype(float)
        out[f"recall@{k}"] = hits / n_target
    prec_at_i = np.cumsum(rel[:, :map_k], axis=1) / np.arange(1, map_k + 1)
    out[f"map@{map_k}"] = (prec_at_i * rel[:, :map_k]).sum(axis=1) / np.minimum(n_target, map_k)
    return pd.DataFrame(out)


def evaluate(model: Recommender, carts: EvalCarts, services: pd.DataFrame,
             ks=(3, 5), map_k: int = 5, exclude: list[str] | None = None) -> pd.DataFrame:
    """Per-cart metrics. `exclude` drops services from both targets and candidates
    (e.g. the $0 dollies); carts left with no target are skipped."""
    ks = list(ks)
    allowed = feasibility_mask(carts.bookings, carts.cart, services)
    target = carts.target.copy()
    if exclude:
        cols = services["service_id"].isin(exclude).to_numpy()
        allowed[:, cols] = False
        target[:, cols] = False
    rows = target.any(axis=1)

    scores = model.score(carts.bookings, carts.cart)
    ranked = top_k(scores[rows], allowed[rows], max(max(ks), map_k))
    m = _row_metrics(ranked, target[rows], ks, map_k)
    m.insert(0, "cart_row", np.flatnonzero(rows))   # row in carts.bookings, for slicing
    m.insert(1, "cart_size", carts.cart[rows].sum(axis=1))
    m.insert(2, "n_hidden", target[rows].sum(axis=1))
    return m


META_COLUMNS = ["cart_row", "cart_size", "n_hidden"]


def summarize(per_cart: pd.DataFrame) -> pd.Series:
    return per_cart.drop(columns=META_COLUMNS).mean()


def by_label(per_cart: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Mean metrics per label, where labels is indexed like carts.bookings (by cart_row)."""
    lab = labels.to_numpy()[per_cart["cart_row"].to_numpy()]
    g = per_cart.drop(columns=META_COLUMNS).groupby(lab)
    out = g.mean()
    out.insert(0, "carts", g.size())
    return out


def head_services(X_train: np.ndarray, services: pd.DataFrame, min_attach: float) -> list[str]:
    """Services bought in at least `min_attach` of training bookings (the 'head').

    Excluding them from targets and candidates gives a long-tail evaluation:
    can the model find the less obvious add-ons?
    """
    return services["service_id"][X_train.mean(axis=0) >= min_attach].tolist()


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of per-cart differences a - b with a 95% bootstrap interval.

    Both models are scored on the same carts, so we resample carts (pairs),
    not the two models independently.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def mean_rank_when_hidden(model: Recommender, carts: EvalCarts, services: pd.DataFrame) -> pd.Series:
    """Per service: average rank (1 = top) among allowed services in carts where it is hidden.

    Shows whether a model moves a niche service up even when it never reaches the top k.
    """
    allowed = feasibility_mask(carts.bookings, carts.cart, services)
    s = np.where(allowed, model.score(carts.bookings, carts.cart), -np.inf)
    order = np.argsort(-s, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(1, s.shape[1] + 1)[None, :].repeat(len(s), 0), axis=1)
    hidden = carts.target.sum(axis=0)
    mean = np.where(carts.target, rank, 0).sum(axis=0) / np.maximum(hidden, 1)
    return pd.Series(mean, index=services["service_id"]).where(hidden > 0)


CART_SIZE_BINS = [-1, 0, 2, 5, np.inf]
CART_SIZE_LABELS = ["0 (empty)", "1-2", "3-5", "6+"]


def by_cart_size(per_cart: pd.DataFrame) -> pd.DataFrame:
    bucket = pd.cut(per_cart["cart_size"], CART_SIZE_BINS, labels=CART_SIZE_LABELS)
    g = per_cart.drop(columns=META_COLUMNS).groupby(bucket, observed=False)
    out = g.mean()
    out.insert(0, "carts", g.size())
    return out
