import numpy as np
import pytest

from moverec.als import ImplicitALS, als_loss, fold_in_lstsq, solve_rows


def block_data(seed=0, n=600):
    """Two groups of services that co-occur; group B is more popular overall."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 8), dtype=bool)
    a_rows = rng.random(n) < 0.3
    X[np.ix_(a_rows, [0, 1, 2])] = rng.random((a_rows.sum(), 3)) < 0.8
    X[np.ix_(~a_rows, [4, 5, 6, 7])] = rng.random(((~a_rows).sum(), 4)) < 0.8
    return X


def test_solve_rows_matches_explicit_normal_equations():
    rng = np.random.default_rng(1)
    R = rng.random((5, 7)) < 0.4
    Y = rng.normal(size=(7, 3))
    alpha, reg = 10.0, 0.5
    got = solve_rows(R, Y, alpha, reg)
    for u in range(5):
        C = np.diag(1 + alpha * R[u])
        A = Y.T @ C @ Y + reg * np.eye(3)
        b = Y.T @ C @ R[u].astype(float)
        np.testing.assert_allclose(got[u], np.linalg.solve(A, b), atol=1e-10)


def test_lstsq_fold_in_equals_batch_fold_in():
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(9, 4))
    carts = rng.random((6, 9)) < 0.3
    batch = solve_rows(carts, Y, 20.0, 1.0)
    for u in range(6):
        np.testing.assert_allclose(fold_in_lstsq(carts[u], Y, 20.0, 1.0), batch[u], atol=1e-8)


def test_loss_never_increases():
    m = ImplicitALS(factors=4, iterations=8, track_loss=True).fit(None, block_data())
    assert all(b <= a + 1e-6 for a, b in zip(m.loss_, m.loss_[1:]))


def test_loss_helper_zero_for_perfect_fit():
    R = np.eye(3, dtype=bool)
    assert als_loss(R, np.eye(3), np.eye(3), alpha=5.0, reg=0.0) == pytest.approx(0.0)


def test_partial_cart_pulls_in_its_group():
    X = block_data()
    m = ImplicitALS(factors=4, reg=0.1, iterations=10).fit(None, X)
    cart = np.zeros((1, 8), dtype=bool)
    cart[0, 0] = True                                 # one item from group A
    s = m.score(None, cart)[0]
    assert min(s[1], s[2]) > max(s[4:])               # A beats the more popular B


def test_empty_cart_falls_back_to_popularity():
    X = block_data()
    m = ImplicitALS(factors=4, iterations=5).fit(None, X)
    s = m.score(None, np.zeros((1, 8), dtype=bool))[0]
    np.testing.assert_array_equal(np.argsort(-s, kind="stable"), np.argsort(-X.mean(0), kind="stable"))


def test_deterministic_given_seed():
    X = block_data()
    a = ImplicitALS(factors=4, iterations=3, seed=7).fit(None, X).item_factors_
    b = ImplicitALS(factors=4, iterations=3, seed=7).fit(None, X).item_factors_
    np.testing.assert_array_equal(a, b)


def test_similar_items_finds_group_mates():
    m = ImplicitALS(factors=4, reg=0.1, iterations=10).fit(None, block_data())
    neighbours = {j for j, _ in m.similar_items(0, k=2)}
    assert neighbours == {1, 2}
