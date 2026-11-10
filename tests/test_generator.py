import copy

import numpy as np
import pandas as pd
import pytest

from moverec.assumptions import AssumptionError, load_assumptions, validate
from moverec.catalog import load_catalog
from moverec.generator import PUBLIC_BOOKING_COLUMNS, generate, write_csvs
from moverec.report import build_report, summary_stats

N = 4000


@pytest.fixture(scope="module")
def a():
    return load_assumptions()


@pytest.fixture(scope="module")
def data(a):
    return generate(a, seed=123, n_bookings=N)


def _basket(data):
    return data.items.merge(data.services, on="service_id")


# ---------------------------------------------------------------- assumptions

def test_every_assumption_has_a_source(a):
    for name, block in a.blocks.items():
        assert block["source"].strip(), name


def test_validation_rejects_missing_source(a):
    raw = {"assumptions": copy.deepcopy(a.blocks)}
    del raw["assumptions"]["need_noise_sigma"]["source"]
    with pytest.raises(AssumptionError):
        validate(raw)


def test_validation_rejects_unverified_flag_missing(a):
    raw = {"assumptions": copy.deepcopy(a.blocks)}
    del raw["assumptions"]["truck_size_by_home_size"]["verified"]
    with pytest.raises(AssumptionError):
        validate(raw)


def test_probability_tables_sum_to_one(a):
    assert sum(a["monthly_move_weights"]) == pytest.approx(1.0)
    assert sum(a["bookings_per_customer"].values()) == pytest.approx(1.0)
    for spec in a["hidden_segments"].values():
        assert sum(spec["home_size"]) == pytest.approx(1.0)
        assert sum(spec["distance"]) == pytest.approx(1.0)
    for key in ("share_start", "share_end"):
        assert sum(s[key] for s in a["hidden_segments"].values()) == pytest.approx(1.0)
    for table in ("truck_size_by_home_size", "storage_size_by_home_size"):
        for row in a[table].values():
            assert sum(row) == pytest.approx(1.0)


# ---------------------------------------------------------------- catalog

def test_catalog_ids_unique_and_complete(a):
    cat = load_catalog(a)
    assert len(cat) == len(set(cat.ids)) == 45
    assert cat.in_group("truck") == ["truck_10ft", "truck_16ft", "truck_26ft"]
    assert len(cat.in_group("storage")) == 5
    assert {s.id for s in cat.services if s.is_free} == {"furniture_dolly", "hand_truck_dolly"}


# ---------------------------------------------------------------- generator

def test_deterministic_given_seed(a):
    d1 = generate(a, seed=7, n_bookings=500)
    d2 = generate(a, seed=7, n_bookings=500)
    pd.testing.assert_frame_equal(d1.bookings, d2.bookings)
    pd.testing.assert_frame_equal(d1.items, d2.items)


def test_different_seeds_differ(a):
    d1 = generate(a, seed=1, n_bookings=500)
    d2 = generate(a, seed=2, n_bookings=500)
    assert not d1.items.equals(d2.items)


def test_hidden_segment_not_in_public_tables(data):
    assert list(data.bookings.columns) == PUBLIC_BOOKING_COLUMNS
    assert "segment" not in data.items.columns


def test_truck_rule(data):
    b = _basket(data)
    trucks = b[b["group"] == "truck"].groupby("booking_id").size()
    truck_moves = set(data.bookings.loc[data.bookings["move_type"] == "truck", "booking_id"])
    labor_only = set(data.bookings["booking_id"]) - truck_moves
    assert trucks.max() == 1
    assert set(trucks.index) == truck_moves          # every truck move has one truck
    assert not (set(trucks.index) & labor_only)      # labor-only moves have none


def test_at_most_one_storage_unit_and_months_positive(data):
    b = _basket(data)
    storage = b[b["group"] == "storage"]
    assert storage.groupby("booking_id").size().max() == 1
    assert (storage["quantity"] >= 1).all()


def test_no_duplicate_line_items_and_positive_quantities(data):
    assert not data.items.duplicated(["booking_id", "service_id"]).any()
    assert (data.items["quantity"] >= 1).all()


def test_dates_are_consistent(data, a):
    b = data.bookings
    assert (b["booking_ts"] <= b["move_date"] + pd.Timedelta(days=1)).all()
    assert (b["lead_time_days"] >= 0).all()
    w = a["simulation_window"]
    assert b["move_date"].min() >= pd.Timestamp(w["start"])
    assert b["move_date"].max() <= pd.Timestamp(w["end"])


def test_mostly_cold_start(data):
    s = summary_stats(data.bookings, data.items)
    assert s["share_customers_single_booking"] > 0.85


def test_repeat_customers_keep_their_segment(data):
    assert not data.hidden_segments["customer_id"].duplicated().any()


def test_summer_busier_than_winter(data):
    month = data.bookings["move_date"].dt.month
    assert month.isin([6, 7, 8]).sum() > 1.5 * month.isin([12, 1, 2]).sum()


def test_stairs_protector_needs_stairs_mostly(data):
    b = data.bookings.set_index("booking_id")
    stairs = (b["origin_access"] == "stairs") | (b["dest_access"] == "stairs")
    sp = set(data.items.loc[data.items["service_id"] == "stairs_protector", "booking_id"])
    with_sp = stairs[stairs.index.isin(sp)]
    assert with_sp.mean() > 0.85  # only basket noise puts it on no-stairs moves


def test_truck_size_follows_home_size(data):
    b = _basket(data)
    t = b[b["group"] == "truck"].merge(data.bookings[["booking_id", "home_size"]], on="booking_id")
    share_26 = (t["service_id"] == "truck_26ft").groupby(t["home_size"]).mean()
    assert share_26["4br_plus"] > share_26["2br"] > share_26["studio"]


def test_every_service_appears(data):
    assert set(data.items["service_id"]) == set(data.services["service_id"])


def test_hidden_segments_leave_a_trace(data):
    # downsizers have a boost for piano that no observed feature explains
    seg = data.bookings.merge(data.hidden_segments, on="customer_id")
    piano = seg["booking_id"].isin(data.items.loc[data.items["service_id"] == "piano", "booking_id"])
    rates = piano.groupby(seg["segment"]).mean()
    assert rates["downsizer"] > 2 * rates.drop("downsizer").mean()


# ---------------------------------------------------------------- outputs

def test_write_csvs_and_report(tmp_path, data, a):
    write_csvs(data, tmp_path)
    for f in ("bookings.csv", "booking_items.csv", "services.csv", "_hidden/customer_segments.csv"):
        assert (tmp_path / f).exists()
    reloaded = pd.read_csv(tmp_path / "bookings.csv", parse_dates=["booking_ts", "move_date"])
    assert len(reloaded) == len(data.bookings)

    text = build_report(data.bookings, data.items, data.services, a, data.hidden_segments)
    assert "synthetic" in text.lower()
    assert "cold start" in text.lower()
    assert "furniture_dolly" in text and "$0" in text
