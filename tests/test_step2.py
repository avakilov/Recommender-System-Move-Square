"""Tests for the split, evaluation harness, baselines and leakage guard."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moverec.assumptions import load_assumptions
from moverec.baselines import CoOccurrence, GlobalPopularity, ProfilePopularity
from moverec.data import basket_matrix, load_dataset
from moverec.evaluate import (_row_metrics, evaluate, feasibility_mask, make_partial_carts,
                              summarize, top_k)
from moverec.generator import generate, write_csvs
from moverec.split import split_summary, time_split

SRC = Path(__file__).resolve().parents[1] / "src" / "moverec"


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    write_csvs(generate(load_assumptions(), seed=5, n_bookings=4000), d)
    return load_dataset(d)


@pytest.fixture(scope="module")
def split(ds):
    return time_split(ds.bookings, "2025-04-01", "2025-07-01")


# ---------------------------------------------------------------- leakage

def test_only_generator_and_report_touch_hidden_segments():
    allowed = {"generator.py", "report.py"}
    for f in SRC.glob("*.py"):
        if f.name in allowed:
            continue
        text = f.read_text()
        assert not re.search(r"\b_hidden\b|hidden_segments|customer_segments", text), f.name


def test_loaded_bookings_have_no_segment(ds):
    assert "segment" not in ds.bookings.columns


# ---------------------------------------------------------------- split

def test_split_is_disjoint_complete_and_ordered(ds, split):
    ids = [set(p["booking_id"]) for p in split.parts().values()]
    assert sum(map(len, ids)) == len(ds.bookings)
    assert not (ids[0] & ids[1]) and not (ids[1] & ids[2])
    assert split.train["booking_ts"].max() < split.val["booking_ts"].min()
    assert split.val["booking_ts"].max() < split.test["booking_ts"].min()


def test_split_summary_cold_start(split):
    s = split_summary(split).set_index("part")
    assert (s["cold_start_share"] > 0.85).all()


def test_split_rejects_bad_dates(ds):
    with pytest.raises(ValueError):
        time_split(ds.bookings, "2025-07-01", "2025-04-01")


# ---------------------------------------------------------------- basket matrix + carts

def test_basket_matrix_matches_items(ds):
    X = basket_matrix(ds.bookings, ds.items, ds.service_ids)
    assert X.sum() == len(ds.items)
    assert X.shape == (len(ds.bookings), len(ds.service_ids))


def test_partial_carts_hide_at_least_one_and_keep_rest(ds, split):
    X = basket_matrix(split.val, ds.items, ds.service_ids)
    c = make_partial_carts(split.val, X, seed=0)
    nonempty = X[X.any(axis=1)]
    assert (c.target.sum(axis=1) >= 1).all()
    assert not (c.cart & c.target).any()
    np.testing.assert_array_equal(c.cart | c.target, nonempty)
    assert (c.cart.sum(axis=1) == 0).any()  # some empty carts are evaluated


def test_partial_carts_deterministic(ds, split):
    X = basket_matrix(split.val, ds.items, ds.service_ids)
    a, b = make_partial_carts(split.val, X, 3), make_partial_carts(split.val, X, 3)
    np.testing.assert_array_equal(a.cart, b.cart)


# ---------------------------------------------------------------- metrics

def test_metrics_hand_computed():
    # targets {0, 2}; ranking [2, 1, 0]
    ranked = np.array([[2, 1, 0]])
    target = np.array([[True, False, True]])
    m = _row_metrics(ranked, target, ks=[1, 3], map_k=3).iloc[0]
    assert m["hit@1"] == 1 and m["recall@1"] == 0.5
    assert m["recall@3"] == 1
    assert m["map@3"] == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_top_k_respects_mask_and_pads():
    scores = np.array([[0.9, 0.8, 0.1]])
    allowed = np.array([[False, True, True]])
    np.testing.assert_array_equal(top_k(scores, allowed, 3), [[1, 2, -1]])


def test_feasibility_rules(ds):
    sids = ds.service_ids
    b = pd.DataFrame({"move_type": ["truck", "labor_only"]})
    cart = np.zeros((2, len(sids)), dtype=bool)
    cart[0, sids.index("truck_16ft")] = True
    cart[0, sids.index("storage_5x5")] = True
    ok = feasibility_mask(b, cart, ds.services)
    trucks = [sids.index(t) for t in ("truck_10ft", "truck_16ft", "truck_26ft")]
    assert not ok[:, trucks].any()                 # 2nd truck / labor-only
    assert not ok[0, sids.index("storage_10x10")]  # 2nd storage unit
    assert ok[1, sids.index("storage_10x10")]


# ---------------------------------------------------------------- baselines

def test_baselines_run_and_beat_nothing_trivial(ds, split):
    X_train = basket_matrix(split.train, ds.items, ds.service_ids)
    carts = make_partial_carts(split.val, basket_matrix(split.val, ds.items, ds.service_ids), 0)
    results = {}
    for model in (GlobalPopularity(), ProfilePopularity(20), CoOccurrence()):
        scores = model.fit(split.train, X_train).score(carts.bookings, carts.cart)
        assert scores.shape == carts.cart.shape and np.isfinite(scores).all()
        results[model.name] = summarize(evaluate(model, carts, ds.services))["map@5"]

    class Random:
        name = "random"
        def score(self, b, cart):
            return np.random.default_rng(0).random(cart.shape)

    rand = summarize(evaluate(Random(), carts, ds.services))["map@5"]
    assert min(results.values()) > rand


def test_profile_popularity_backs_off_to_global(ds, split):
    X = basket_matrix(split.train, ds.items, ds.service_ids)
    m = ProfilePopularity(20).fit(split.train, X)
    unseen = {"move_type": "spaceship", "home_size": "castle", "distance_band": "orbit"}
    np.testing.assert_allclose(m.rate_for(unseen), X.mean(axis=0))


def test_co_occurrence_empty_cart_equals_popularity_order(ds, split):
    X = basket_matrix(split.train, ds.items, ds.service_ids)
    co = CoOccurrence().fit(split.train, X)
    s = co.score(split.val.head(1), np.zeros((1, X.shape[1]), dtype=bool))[0]
    np.testing.assert_array_equal(np.argsort(-s, kind="stable"), np.argsort(-X.mean(0), kind="stable"))


def test_co_occurrence_uses_cart():
    X = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=bool)
    co = CoOccurrence().fit(None, X)
    s = co.score(None, np.array([[True, False, False]]))[0]
    assert s[1] > s[2]  # 1 co-occurs with 0 although 2 is more popular


# ---------------------------------------------------------------- significance + ranks

def test_paired_bootstrap():
    from moverec.evaluate import paired_bootstrap
    a = np.array([0.5, 0.6, 0.7, 0.8] * 50)
    r = paired_bootstrap(a, a - 0.1)
    assert r["diff"] == pytest.approx(0.1) and r["ci_low"] == pytest.approx(0.1)
    e = np.random.default_rng(0).random(len(a) // 2)
    noisy = paired_bootstrap(a, a + np.concatenate([e, -e]))   # true mean difference 0
    assert noisy["diff"] == pytest.approx(0.0, abs=1e-12)
    assert noisy["ci_low"] < 0 < noisy["ci_high"]


def test_mean_rank_when_hidden():
    from moverec.evaluate import EvalCarts, mean_rank_when_hidden
    services = pd.DataFrame({"service_id": ["a", "b", "c"], "group": ["x", "x", "x"]})

    class Fixed:
        def score(self, b, cart):
            return np.tile([3.0, 2.0, 1.0], (len(cart), 1))

    carts = EvalCarts(pd.DataFrame({"move_type": ["truck", "truck"]}),
                      cart=np.array([[True, False, False], [False, False, False]]),
                      target=np.array([[False, False, True], [False, True, False]]))
    r = mean_rank_when_hidden(Fixed(), carts, services)
    assert r["c"] == 2 and r["b"] == 2 and np.isnan(r["a"])


# ---------------------------------------------------------------- step 6 helpers

def test_cart_row_points_back_to_bookings(ds, split):
    from moverec.evaluate import by_label, head_services
    X = basket_matrix(split.val, ds.items, ds.service_ids)
    carts = make_partial_carts(split.val, X, seed=0)
    free = ds.services.loc[ds.services["is_free"], "service_id"].tolist()
    per = evaluate(GlobalPopularity().fit(split.train, basket_matrix(split.train, ds.items, ds.service_ids)),
                   carts, ds.services, exclude=free)
    # rows dropped by the exclusion are skipped, the rest keep their position
    assert per["cart_row"].is_monotonic_increasing
    assert (carts.target[per["cart_row"]].sum(axis=1) >= per["n_hidden"]).all()

    labels = pd.Series(np.where(carts.bookings["move_type"] == "truck", "truck", "labor"))
    sl = by_label(per, labels)
    assert sl["carts"].sum() == len(per)

    head = head_services(X, ds.services, min_attach=0.2)
    assert set(head) == set(np.array(ds.service_ids)[X.mean(0) >= 0.2])
