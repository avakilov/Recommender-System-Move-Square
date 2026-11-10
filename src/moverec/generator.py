"""Synthetic moving-booking generator.

ALL OUTPUT IS SYNTHETIC. Every number comes from config/assumptions.yaml.

How one booking is made:
  1. A customer gets a move date (seasonal) and a HIDDEN segment (drifts over
     time). Most customers book once; a few book again months later.
  2. The segment drives the observed move profile: home size, distance,
     labor-only vs truck, lead time. Access (stairs/elevator) is random.
  3. Five latent "needs" (diy_pack, pro_pack, protect, heavy, store) are
     segment offset + profile effects + noise.
  4. Each service is bought with probability sigmoid(logit), where the logit
     combines needs, segment-only boosts, direct profile effects, items already
     in the basket, and noise. Trucks and storage units are picked from size
     tables keyed on home size.
  5. Basket noise adds a random service or drops one, creating rare combos.

Models must never read the hidden segment file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .assumptions import Assumptions, load_assumptions
from .catalog import STORAGE_GROUP, TRUCK_GROUP, Catalog, load_catalog

HOME_SIZES = ["studio", "1br", "2br", "3br", "4br_plus"]
DISTANCE_BANDS = ["local", "regional", "long_distance"]
ACCESS_TYPES = ["ground", "stairs", "elevator"]
NEEDS = ["diy_pack", "pro_pack", "protect", "heavy", "store"]
SEASON_OF_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}
STORAGE_KEY = "storage"  # propensity / boost key for the whole storage group

PUBLIC_BOOKING_COLUMNS = [
    "booking_id", "customer_id", "booking_ts", "move_date", "lead_time_days",
    "move_type", "home_size", "distance_band", "distance_miles",
    "origin_access", "dest_access",
]

# Named boolean conditions on observed booking features (used in the YAML).
CONDITIONS = {
    "short_lead_time": lambda b, a: b["lead_time_days"] <= a["lead_time"]["short_lead_days"],
    "labor_only": lambda b, a: b["move_type"] == "labor_only",
    "regional": lambda b, a: b["distance_band"] == "regional",
    "long_distance": lambda b, a: b["distance_band"] == "long_distance",
    "summer": lambda b, a: b["move_date"].dt.month.map(SEASON_OF_MONTH) == "summer",
    "winter": lambda b, a: b["move_date"].dt.month.map(SEASON_OF_MONTH) == "winter",
    "stairs_any": lambda b, a: (b["origin_access"] == "stairs") | (b["dest_access"] == "stairs"),
    "no_stairs": lambda b, a: (b["origin_access"] != "stairs") & (b["dest_access"] != "stairs"),
}


@dataclass
class SyntheticData:
    bookings: pd.DataFrame          # observed booking profile, one row per booking
    items: pd.DataFrame             # booking_id, service_id, quantity
    services: pd.DataFrame          # the catalog
    hidden_segments: pd.DataFrame   # customer_id, segment  -- NEVER a model input


# ----------------------------------------------------------------- helpers

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _categorical(rng: np.random.Generator, probs: np.ndarray) -> np.ndarray:
    """Sample one index per row of a (n, k) probability matrix."""
    cum = np.cumsum(probs, axis=1)
    u = rng.random(len(probs)) * cum[:, -1]
    return np.minimum((cum < u[:, None]).sum(axis=1), probs.shape[1] - 1)


def _feature(name: str, bookings: pd.DataFrame, a: Assumptions) -> np.ndarray:
    if name == "home_size_centered":
        return bookings["home_size_idx"].to_numpy() - 2.0
    return CONDITIONS[name](bookings, a).to_numpy().astype(float)


def _need_vector(d: dict) -> np.ndarray:
    unknown = set(d) - set(NEEDS)
    if unknown:
        raise ValueError(f"unknown needs: {sorted(unknown)}")
    return np.array([d.get(k, 0.0) for k in NEEDS])


def check_config(a: Assumptions, catalog: Catalog) -> None:
    """Fail early if the YAML refers to unknown services or conditions."""
    sampled = [s.id for s in catalog.services if s.group not in (TRUCK_GROUP, STORAGE_GROUP)]
    expected = set(sampled) | {STORAGE_KEY}
    prop = set(a["service_propensity"])
    if prop != expected:
        raise ValueError(f"service_propensity mismatch: missing {sorted(expected - prop)}, "
                         f"extra {sorted(prop - expected)}")
    for seg, boosts in a["segment_service_boosts"].items():
        if seg not in a["hidden_segments"]:
            raise ValueError(f"boosts for unknown segment {seg}")
        if set(boosts) - expected:
            raise ValueError(f"{seg}: boosts for unknown services {sorted(set(boosts) - expected)}")
    for e in a["feature_effects"]:
        if e["service"] not in expected or e["condition"] not in CONDITIONS:
            raise ValueError(f"bad feature effect {e}")
    for feat in a["need_feature_effects"]:
        if feat != "home_size_centered" and feat not in CONDITIONS:
            raise ValueError(f"unknown need feature {feat}")
    # conditional effects may only depend on services sampled earlier
    position = {sid: i for i, sid in enumerate(catalog.ids)}
    position[STORAGE_KEY] = position[catalog.in_group(STORAGE_GROUP)[0]]
    for e in a["conditional_effects"]:
        for dep in e["if_any"]:
            if position[dep] >= position[e["then"]]:
                raise ValueError(f"{e['then']} depends on later service {dep}")


# ----------------------------------------------------------------- stages

def _sample_move_dates(rng, n, start, end, month_weights) -> pd.DatetimeIndex:
    days = pd.date_range(start, end, freq="D")
    w = np.asarray(month_weights)[days.month - 1] / days.days_in_month.to_numpy()
    return days[rng.choice(len(days), size=n, p=w / w.sum())]


def _sample_segments(rng, a, dates: pd.DatetimeIndex, start, end) -> np.ndarray:
    segs = a["hidden_segments"]
    names = list(segs)
    frac = np.asarray((dates - start) / (end - start), dtype=float)[:, None]
    s0 = np.array([segs[n]["share_start"] for n in names])
    s1 = np.array([segs[n]["share_end"] for n in names])
    shares = s0 * (1 - frac) + s1 * frac
    seasons = dates.month.map(SEASON_OF_MONTH)
    for j, n in enumerate(names):
        for season, mult in segs[n].get("season_multipliers", {}).items():
            shares[np.asarray(seasons == season), j] *= mult
    return np.array(names)[_categorical(rng, shares)]


def _sample_customers(rng, a, start, end, target) -> pd.DataFrame:
    k_probs = a["bookings_per_customer"]
    k_vals = np.array(sorted(k_probs))
    p = np.array([k_probs[k] for k in k_vals])
    n_cust = int(round(target / float(k_vals @ p)))

    first = _sample_move_dates(rng, n_cust, start, end, a["monthly_move_weights"])
    segment = _sample_segments(rng, a, first, start, end)
    n_bookings = rng.choice(k_vals, size=n_cust, p=p)
    gap = a["repeat_gap_days"]

    rows = []
    for c in range(n_cust):
        date = first[c]
        for j in range(n_bookings[c]):
            if j > 0:
                date = date + pd.Timedelta(days=int(gap["min_days"] + rng.exponential(gap["extra_mean_days"])))
                if date > end:
                    break
            rows.append((f"C{c:06d}", segment[c], date))
    return pd.DataFrame(rows, columns=["customer_id", "segment", "move_date"])


def _sample_profiles(rng, a, df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    home = np.zeros(n, dtype=int)
    dist = np.zeros(n, dtype=int)
    labor = np.zeros(n, dtype=bool)
    lead_median = np.zeros(n)
    for name, spec in a["hidden_segments"].items():
        m = (df["segment"] == name).to_numpy()
        k = int(m.sum())
        home[m] = rng.choice(len(HOME_SIZES), size=k, p=spec["home_size"])
        dist[m] = rng.choice(len(DISTANCE_BANDS), size=k, p=spec["distance"])
        labor[m] = rng.random(k) < spec["labor_only_prob"]
        lead_median[m] = spec["lead_time_median_days"]

    lt = a["lead_time"]
    is_long = dist == DISTANCE_BANDS.index("long_distance")
    lead = lead_median * np.where(is_long, lt["long_distance_multiplier"], 1.0)
    lead = np.clip(np.round(lead * rng.lognormal(0.0, lt["lognormal_sigma"], n)), 0, lt["max_days"]).astype(int)

    bands = a["distance_band_miles"]
    lo = np.array([bands[b][0] for b in DISTANCE_BANDS])[dist]
    hi = np.array([bands[b][1] for b in DISTANCE_BANDS])[dist]
    miles = np.round(np.exp(rng.uniform(np.log(lo), np.log(hi)))).astype(int)

    acc = a["access_type_probs"]
    acc_p = [acc[t] for t in ACCESS_TYPES]

    out = df.copy()
    out["home_size_idx"] = home
    out["home_size"] = np.array(HOME_SIZES)[home]
    out["distance_band"] = np.array(DISTANCE_BANDS)[dist]
    out["distance_miles"] = miles
    out["move_type"] = np.where(labor, "labor_only", "truck")
    out["lead_time_days"] = lead
    out["origin_access"] = rng.choice(ACCESS_TYPES, size=n, p=acc_p)
    out["dest_access"] = rng.choice(ACCESS_TYPES, size=n, p=acc_p)
    minutes = rng.integers(7 * 60, 22 * 60, size=n)  # booked between 07:00 and 22:00
    out["booking_ts"] = (out["move_date"] - pd.to_timedelta(lead, unit="D")
                         + pd.to_timedelta(minutes, unit="m"))

    out = out.sort_values(["booking_ts", "customer_id"]).reset_index(drop=True)
    out.insert(0, "booking_id", [f"B{i:06d}" for i in range(len(out))])
    return out


def _sample_needs(rng, a, bookings: pd.DataFrame) -> np.ndarray:
    n = len(bookings)
    needs = np.zeros((n, len(NEEDS)))
    for seg, offsets in a["segment_need_offsets"].items():
        needs[(bookings["segment"] == seg).to_numpy()] += _need_vector(offsets)
    for feat, effects in a["need_feature_effects"].items():
        needs += _feature(feat, bookings, a)[:, None] * _need_vector(effects)[None, :]
    return needs + rng.normal(0.0, a["need_noise_sigma"], needs.shape)


def _sample_items(rng, a, catalog: Catalog, bookings: pd.DataFrame, needs: np.ndarray) -> np.ndarray:
    """Return a (n_bookings, n_services) boolean basket matrix."""
    n = len(bookings)
    X = np.zeros((n, len(catalog)), dtype=bool)
    segment = bookings["segment"].to_numpy()
    home_idx = bookings["home_size_idx"].to_numpy()
    storage_ids = catalog.in_group(STORAGE_GROUP)

    def expand(ids):
        return [x for i in ids for x in (storage_ids if i == STORAGE_KEY else [i])]

    def logit(key: str) -> np.ndarray:
        spec = a["service_propensity"][key]
        z = spec["base"] + needs @ _need_vector(spec["loads"])
        for seg, boosts in a["segment_service_boosts"].items():
            z[segment == seg] += boosts.get(key, 0.0)
        for e in a["feature_effects"]:
            if e["service"] == key:
                z += e["delta"] * _feature(e["condition"], bookings, a)
        for e in a["conditional_effects"]:
            if e["then"] == key:
                cols = [catalog.index[s] for s in expand(e["if_any"])]
                z += e["delta"] * X[:, cols].any(axis=1)
        return z + rng.normal(0.0, a["item_noise_sigma"], n)

    # Truck: exactly one for truck moves, size from the home-size table.
    trucks = catalog.in_group(TRUCK_GROUP)
    truck_table = np.array([a["truck_size_by_home_size"][h] for h in HOME_SIZES])
    truck_pick = _categorical(rng, truck_table[home_idx])
    is_truck_move = (bookings["move_type"] == "truck").to_numpy()
    for j, sid in enumerate(trucks):
        X[is_truck_move & (truck_pick == j), catalog.index[sid]] = True

    storage_done = False
    for s in catalog.services:
        if s.group == TRUCK_GROUP:
            continue
        if s.group == STORAGE_GROUP:
            if storage_done:
                continue
            has = rng.random(n) < _sigmoid(logit(STORAGE_KEY))
            table = np.array([a["storage_size_by_home_size"][h] for h in HOME_SIZES])
            size = _categorical(rng, table[home_idx])
            for j, sid in enumerate(storage_ids):
                X[has & (size == j), catalog.index[sid]] = True
            storage_done = True
            continue
        X[:, catalog.index[s.id]] = rng.random(n) < _sigmoid(logit(s.id))
    return X


def _apply_basket_noise(rng, a, catalog: Catalog, X: np.ndarray) -> np.ndarray:
    X = X.copy()
    noise = a["basket_noise"]
    n = len(X)
    plain = np.array([i for i, s in enumerate(catalog.services)
                      if s.group not in (TRUCK_GROUP, STORAGE_GROUP)])
    droppable = np.array([s.group != TRUCK_GROUP for s in catalog.services])

    add_rows = np.flatnonzero(rng.random(n) < noise["random_add_prob"])
    X[add_rows, rng.choice(plain, size=len(add_rows))] = True

    for r in np.flatnonzero(rng.random(n) < noise["random_drop_prob"]):
        cols = np.flatnonzero(X[r] & droppable)
        if cols.size:
            X[r, rng.choice(cols)] = False
    return X


def _quantities(rng, a, catalog: Catalog, bookings: pd.DataFrame, X: np.ndarray) -> np.ndarray:
    Q = X.astype(int)
    q = a["quantities"]
    home_idx = bookings["home_size_idx"].to_numpy()
    box_mean = np.asarray(q["boxes_total_mean_by_home_size"], dtype=float)[home_idx]
    for sid, share in q["box_split"].items():
        i = catalog.index[sid]
        Q[:, i] = X[:, i] * np.maximum(1, rng.poisson(box_mean * share))
    for sid in catalog.in_group("tape"):
        i = catalog.index[sid]
        Q[:, i] = X[:, i] * (1 + rng.poisson(q["tape_rolls_per_home_size_step"] * home_idx))
    sm = a["storage_months"]
    for sid in catalog.in_group(STORAGE_GROUP):
        i = catalog.index[sid]
        months = np.minimum(rng.geometric(sm["geometric_p"], size=len(X)), sm["max_months"])
        Q[:, i] = X[:, i] * months
    return Q


# ----------------------------------------------------------------- public API

def generate(assumptions: Assumptions | None = None, seed: int = 0,
             n_bookings: int | None = None) -> SyntheticData:
    a = assumptions or load_assumptions()
    catalog = load_catalog(a)
    check_config(a, catalog)
    rng = np.random.default_rng(seed)

    window = a["simulation_window"]
    start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
    target = n_bookings or window["target_bookings"]

    bookings = _sample_customers(rng, a, start, end, target)
    bookings = _sample_profiles(rng, a, bookings)
    needs = _sample_needs(rng, a, bookings)
    X = _sample_items(rng, a, catalog, bookings, needs)
    X = _apply_basket_noise(rng, a, catalog, X)
    Q = _quantities(rng, a, catalog, bookings, X)

    rows, cols = np.nonzero(X)
    items = pd.DataFrame({
        "booking_id": bookings["booking_id"].to_numpy()[rows],
        "service_id": np.array(catalog.ids)[cols],
        "quantity": Q[rows, cols],
    })
    hidden = (bookings[["customer_id", "segment"]].drop_duplicates()
              .sort_values("customer_id").reset_index(drop=True))
    return SyntheticData(
        bookings=bookings[PUBLIC_BOOKING_COLUMNS].copy(),
        items=items,
        services=catalog.to_frame(),
        hidden_segments=hidden,
    )


HIDDEN_README = """\
This folder holds the generator's HIDDEN ground truth (customer segments).
It exists only for diagnostics. No model, feature, split or tuning code may
read it. Doing so would make the evaluation circular.
"""


def write_csvs(data: SyntheticData, out_dir: str | Path) -> None:
    out = Path(out_dir)
    (out / "_hidden").mkdir(parents=True, exist_ok=True)
    data.bookings.to_csv(out / "bookings.csv", index=False)
    data.items.to_csv(out / "booking_items.csv", index=False)
    data.services.to_csv(out / "services.csv", index=False)
    data.hidden_segments.to_csv(out / "_hidden" / "customer_segments.csv", index=False)
    (out / "_hidden" / "README.txt").write_text(HIDDEN_README)
