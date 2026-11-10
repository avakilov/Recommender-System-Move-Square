import numpy as np
import pandas as pd
import pytest

from moverec.content import GROUPS, ContentKNN, ProfileEncoder


def profiles(n, rng, **fixed):
    df = pd.DataFrame({
        "move_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "home_size": rng.choice(["studio", "1br", "2br", "3br", "4br_plus"], n),
        "move_type": rng.choice(["truck", "labor_only"], n),
        "distance_miles": rng.integers(2, 2000, n),
        "lead_time_days": rng.integers(0, 60, n),
        "origin_access": rng.choice(["ground", "stairs", "elevator"], n),
        "dest_access": rng.choice(["ground", "stairs", "elevator"], n),
    })
    for col, val in fixed.items():
        df[col] = val
    return df


def test_encoder_balances_groups():
    b = profiles(2000, np.random.default_rng(0))
    Z = ProfileEncoder().fit(b).transform(b)
    widths = {"origin_access": 3, "dest_access": 3, "move_season": 2}
    col = 0
    for g in GROUPS:
        w = widths.get(g, 1)
        assert Z[:, col:col + w].var(axis=0).sum() == pytest.approx(1.0, rel=1e-6)
        col += w
    assert col == Z.shape[1]


def test_encoder_group_weight_and_validation():
    b = profiles(500, np.random.default_rng(1))
    plain = ProfileEncoder().fit(b).transform(b)
    heavy = ProfileEncoder({"home_size": 3.0}).fit(b).transform(b)
    np.testing.assert_allclose(heavy[:, 0], 3 * plain[:, 0])
    with pytest.raises(ValueError):
        ProfileEncoder({"not_a_group": 1.0})


def test_december_is_near_january():
    rng = np.random.default_rng(2)
    b = profiles(3, rng, home_size="2br", move_type="truck", distance_miles=10,
                 lead_time_days=7, origin_access="ground", dest_access="ground")
    b["move_date"] = pd.to_datetime(["2024-12-15", "2025-01-15", "2025-07-15"])
    train = profiles(500, rng)
    Z = ProfileEncoder().fit(train).transform(b)
    assert np.linalg.norm(Z[0] - Z[1]) < np.linalg.norm(Z[0] - Z[2])


def test_knn_recommends_what_similar_moves_bought():
    rng = np.random.default_rng(3)
    big = profiles(300, rng, home_size="4br_plus", distance_miles=1500)
    small = profiles(300, rng, home_size="studio", distance_miles=5)
    train = pd.concat([big, small], ignore_index=True)
    X = np.zeros((600, 3), dtype=bool)
    X[:300, 0] = True             # big moves buy service 0
    X[300:, 1] = True             # small moves buy service 1
    X[:, 2] = rng.random(600) < 0.9   # service 2 popular everywhere
    m = ContentKNN(k=50).fit(train, X)
    q = profiles(1, rng, home_size="4br_plus", distance_miles=1400)
    s = m.score(q, np.zeros((1, 3), dtype=bool))[0]
    assert s[0] > s[1]


def test_batching_does_not_change_results():
    rng = np.random.default_rng(4)
    train, query = profiles(800, rng), profiles(100, rng)
    X = rng.random((800, 6)) < 0.3
    a = ContentKNN(k=30, batch_size=7).fit(train, X).score(query, np.zeros((100, 6), bool))
    b = ContentKNN(k=30, batch_size=1000).fit(train, X).score(query, np.zeros((100, 6), bool))
    np.testing.assert_allclose(a, b)


def test_exact_profile_match_is_top_neighbour():
    rng = np.random.default_rng(5)
    train = profiles(400, rng)
    m = ContentKNN(k=5).fit(train, rng.random((400, 4)) < 0.5)
    idx, sim = m.neighbours(train.iloc[[17]])
    assert sim[0, 0] == pytest.approx(1.0)
    assert np.allclose(m.train_vecs_[idx[0, 0]], m.train_vecs_[17])


def test_scores_are_rates():
    rng = np.random.default_rng(6)
    train = profiles(400, rng)
    m = ContentKNN(k=20).fit(train, rng.random((400, 5)) < 0.4)
    s = m.score(profiles(50, rng), np.zeros((50, 5), bool))
    assert (s >= 0).all() and (s <= 1 + 1e-5).all()
