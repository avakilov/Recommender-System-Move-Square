"""Implicit-feedback ALS on the booking x service matrix, in plain NumPy.

Model (Hu, Koren & Volinsky 2008):
    p_ui = 1 if booking u bought service i, else 0
    c_ui = 1 + alpha * p_ui                         (confidence)
    minimise  sum_ui c_ui (p_ui - x_u . y_i)^2 + reg * (|X|^2 + |Y|^2)

Each half-step is an exact ridge solve. For one row u:
    (Y'Y + alpha * sum_{i in u} y_i y_i' + reg*I) x_u = (1 + alpha) * sum_{i in u} y_i

Because almost every booking is a new customer, the booking factors learned in
training are thrown away. At recommendation time a partial cart is
"folded in": we solve the same least-squares problem for a fresh row with
the item factors held fixed. An empty cart folds in to x = 0 and carries no
signal; there we fall back to popularity (the content model in step 4 is
meant to cover that case).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def solve_rows(R: np.ndarray, Y: np.ndarray, alpha: float, reg: float) -> np.ndarray:
    """Solve every row of R against fixed factors Y at once. R: (n, m) 0/1, Y: (m, f)."""
    R = R.astype(float)
    m, f = Y.shape
    outer = np.einsum("mf,mg->mfg", Y, Y).reshape(m, f * f)        # y_i y_i' per column
    A = Y.T @ Y + alpha * (R @ outer).reshape(-1, f, f) + reg * np.eye(f)
    b = (1.0 + alpha) * (R @ Y)
    return np.linalg.solve(A, b[..., None])[..., 0]


def fold_in_lstsq(cart: np.ndarray, Y: np.ndarray, alpha: float, reg: float) -> np.ndarray:
    """Fold in ONE cart with np.linalg.lstsq (readable reference version).

    Weighted ridge regression written as ordinary least squares:
        rows sqrt(c_i) * y_i  -> target sqrt(c_i) * p_i     (one per service)
        rows sqrt(reg) * I    -> target 0                   (the ridge penalty)
    Gives the same answer as solve_rows on a single row.
    """
    p = cart.astype(float)
    w = np.sqrt(1.0 + alpha * p)
    f = Y.shape[1]
    A = np.vstack([w[:, None] * Y, np.sqrt(reg) * np.eye(f)])
    b = np.concatenate([w * p, np.zeros(f)])
    return np.linalg.lstsq(A, b, rcond=None)[0]


def als_loss(R: np.ndarray, U: np.ndarray, Y: np.ndarray, alpha: float, reg: float) -> float:
    P = R.astype(float)
    C = 1.0 + alpha * P
    return float((C * (P - U @ Y.T) ** 2).sum() + reg * ((U ** 2).sum() + (Y ** 2).sum()))


class ImplicitALS:
    name = "implicit_als"

    def __init__(self, factors: int = 16, reg: float = 1.0, alpha: float = 20.0,
                 iterations: int = 15, seed: int = 0, track_loss: bool = False):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed, self.track_loss = iterations, seed, track_loss

    def fit(self, bookings: pd.DataFrame, X: np.ndarray) -> "ImplicitALS":
        rng = np.random.default_rng(self.seed)
        Y = rng.normal(0.0, 0.1, (X.shape[1], self.factors))
        self.loss_ = []
        for _ in range(self.iterations):
            U = solve_rows(X, Y, self.alpha, self.reg)
            Y = solve_rows(X.T, U, self.alpha, self.reg)
            if self.track_loss:
                self.loss_.append(als_loss(X, U, Y, self.alpha, self.reg))
        self.item_factors_ = Y
        self.popularity_ = X.mean(axis=0)
        return self

    def fold_in(self, cart: np.ndarray) -> np.ndarray:
        """Booking factors for a batch of partial carts (n, n_services) -> (n, factors)."""
        return solve_rows(cart, self.item_factors_, self.alpha, self.reg)

    def score(self, bookings: pd.DataFrame, cart: np.ndarray) -> np.ndarray:
        # empty carts give zero scores; the tiny popularity term orders them
        return self.fold_in(cart) @ self.item_factors_.T + 1e-6 * self.popularity_

    def similar_items(self, service_index: int, k: int = 5) -> list[tuple[int, float]]:
        """Nearest services by cosine similarity of item factors (for inspection)."""
        Y = self.item_factors_
        Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        sims = Yn @ Yn[service_index]
        sims[service_index] = -np.inf
        top = np.argsort(-sims)[:k]
        return [(int(j), float(sims[j])) for j in top]
