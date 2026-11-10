import numpy as np
import pandas as pd
import pytest

from moverec.baselines import CoOccurrenceLift
from moverec.hybrid import (FixedScores, Hybrid, combine, component_scores, row_standardize,
                            tune_weights, weight_grid)


class Const:
    """A component that returns fixed scores regardless of input."""
    def __init__(self, row):
        self.row = np.asarray(row, float)

    def score(self, bookings, cart):
        return np.tile(self.row, (len(cart), 1))


def test_row_standardize_handles_flat_rows():
    S = np.array([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]])
    Z = row_standardize(S)
    np.testing.assert_allclose(Z[0], [-1.2247449, 0.0, 1.2247449], rtol=1e-6)
    np.testing.assert_array_equal(Z[1], 0.0)


def test_lift_zero_for_independent_positive_for_associated():
    rng = np.random.default_rng(0)
    n = 20000
    a = rng.random(n) < 0.5
    b = rng.random(n) < 0.5                              # independent of a
    c = np.where(a, rng.random(n) < 0.8, rng.random(n) < 0.2)   # follows a
    X = np.column_stack([a, b, c])
    m = CoOccurrenceLift(prior=10).fit(None, X)
    s = m.score(None, np.array([[True, False, False]]))[0]
    assert abs(s[1]) < 0.05
    assert s[2] > 0.4


def test_lift_empty_cart_is_zero_and_rare_item_shrinks():
    X = np.zeros((1000, 3), dtype=bool)
    X[:500, 1] = True
    X[0, [0, 2]] = True                 # item 0 seen once, with item 2
    m = CoOccurrenceLift(prior=10).fit(None, X)
    np.testing.assert_array_equal(m.score(None, np.zeros((1, 3), bool)), 0.0)
    unshrunk = np.log(1.0 / X[:, 2].mean())
    assert 0 < m.log_lift_[0, 2] < unshrunk


def test_single_component_hybrid_ranks_like_that_component():
    comps = {"profile": Const([0.5, 0.1, 0.3]), "cooc_lift": Const([0.0, 9.0, 0.0])}
    h = Hybrid(comps, {"profile": 1.0, "cooc_lift": 0.0})
    s = h.score(None, np.zeros((2, 3), bool))
    np.testing.assert_array_equal(np.argsort(-s, axis=1)[0], [0, 2, 1])


def test_flat_component_abstains():
    comps = {"profile": Const([0.5, 0.1, 0.3]), "als": Const([1.0, 1.0, 1.0])}
    s1 = Hybrid(comps, {"profile": 1.0, "als": 5.0}).score(None, np.zeros((1, 3), bool))
    s2 = Hybrid(comps, {"profile": 1.0}).score(None, np.zeros((1, 3), bool))
    np.testing.assert_allclose(s1, s2)


def test_only_weight_ratios_matter():
    scores = component_scores({"profile": Const([0.5, 0.1, 0.3]), "als": Const([0.0, 2.0, 1.0])},
                              None, np.zeros((1, 3), bool))
    a = combine(scores, {"profile": 1.0, "als": 0.5})
    b = combine(scores, {"profile": 4.0, "als": 2.0})
    np.testing.assert_array_equal(np.argsort(-a), np.argsort(-b))


def test_validation():
    with pytest.raises(ValueError):
        Hybrid({"profile": Const([1, 2])}, {"nope": 1.0})
    with pytest.raises(ValueError):
        combine({"profile": np.ones((1, 2))}, {"profile": 0.0})


def test_weight_grid_skips_all_zero_and_respects_fixed():
    grid = list(weight_grid(["a", "b"], [0, 1], fixed_zero=("b",)))
    assert grid == [{"a": 1, "b": 0.0}]
    assert len(list(weight_grid(["a", "b", "c"], [0, 1, 2]))) == 26


def test_tune_weights_finds_the_informative_component():
    target = np.array([[0, 1, 0]] * 4, dtype=float)
    scores = {"good": row_standardize(target + 0.01 * np.arange(3)),
              "bad": row_standardize(np.tile([3.0, 0.0, 1.0], (4, 1)))}
    w, metric, table = tune_weights(scores, lambda s: float((s.argmax(1) == 1).mean()), [0, 1, 2])
    assert w["bad"] == 0 and w["good"] > 0 and metric == 1.0
    assert len(table) == 8


def test_fixed_scores_passthrough():
    m = np.eye(3)
    assert FixedScores(m).score(pd.DataFrame(), None) is m
