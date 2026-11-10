"""Short data-quality report for the synthetic dataset (Markdown)."""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from .assumptions import Assumptions
from .generator import SEASON_OF_MONTH


def _table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def summary_stats(bookings: pd.DataFrame, items: pd.DataFrame) -> dict:
    """Headline numbers (also used by tests)."""
    per_customer = bookings.groupby("customer_id").size()
    ordered = bookings.sort_values("booking_ts")
    is_first = ~ordered.duplicated("customer_id")
    size = items.groupby("booking_id").size().reindex(bookings["booking_id"], fill_value=0)
    return {
        "n_bookings": len(bookings),
        "n_customers": int(per_customer.size),
        "share_customers_single_booking": float((per_customer == 1).mean()),
        "share_bookings_first_time": float(is_first.mean()),
        "services_per_booking_mean": float(size.mean()),
        "services_per_booking_median": float(size.median()),
        "share_empty_baskets": float((size == 0).mean()),
    }


def build_report(bookings: pd.DataFrame, items: pd.DataFrame, services: pd.DataFrame,
                 assumptions: Assumptions, hidden_segments: pd.DataFrame | None = None) -> str:
    s = summary_stats(bookings, items)
    b = bookings.copy()
    b["move_date"] = pd.to_datetime(b["move_date"])
    b["booking_ts"] = pd.to_datetime(b["booking_ts"])
    b["move_month"] = b["move_date"].dt.to_period("M").astype(str)
    b["season"] = b["move_date"].dt.month.map(SEASON_OF_MONTH)
    out = []

    out.append("# Data-quality report: SYNTHETIC moving bookings\n")
    out.append("> **Every row here is synthetic**, generated from `config/assumptions.yaml`. "
               "These numbers describe the simulator, not any real business.\n")

    # --- volume and cold start
    out.append("## Volume and cold start\n")
    out.append(_table(pd.DataFrame([
        ("bookings", f"{s['n_bookings']:,}"),
        ("customers", f"{s['n_customers']:,}"),
        ("booking dates", f"{b['booking_ts'].min():%Y-%m-%d} to {b['booking_ts'].max():%Y-%m-%d}"),
        ("move dates", f"{b['move_date'].min():%Y-%m-%d} to {b['move_date'].max():%Y-%m-%d}"),
        ("customers with exactly one booking", _pct(s["share_customers_single_booking"])),
        ("bookings with no prior customer history (cold start)", _pct(s["share_bookings_first_time"])),
    ], columns=["metric", "value"])))
    out.append("\nBooking dates can fall before the window start because a move early "
               "in the window may have been booked weeks earlier.\n")

    # --- basket size
    size = items.groupby("booking_id").size().reindex(b["booking_id"], fill_value=0)
    out.append("## Services per booking\n")
    out.append(f"Mean {s['services_per_booking_mean']:.2f}, median {s['services_per_booking_median']:.0f}, "
               f"90th percentile {size.quantile(0.9):.0f}, max {size.max()}. "
               f"Empty baskets: {_pct(s['share_empty_baskets'])}.\n")
    dist = size.clip(upper=12).value_counts().sort_index()
    out.append(_table(pd.DataFrame({
        "services": [f"{k}+" if k == 12 else str(k) for k in dist.index],
        "bookings": dist.values,
        "share": [_pct(v / len(size)) for v in dist.values],
    })))
    out.append("")

    # --- attach rates
    attach = (items.groupby("service_id")["booking_id"].nunique() / len(b)).rename("attach")
    cat = services.set_index("service_id").join(attach).fillna({"attach": 0.0})
    cat = cat.sort_values("attach", ascending=False).reset_index()
    out.append("## Attach rate per service\n")
    out.append("Attach rate = share of all bookings that include the service. "
               "`free` = $0 line item (adds no revenue; recommend only as a reminder). "
               "`rare` = under 1% of bookings.\n")
    cat["flags"] = [", ".join(f for f, on in (("free", fr), ("rare", a < 0.01)) if on)
                    for fr, a in zip(cat["is_free"], cat["attach"])]
    out.append(_table(pd.DataFrame({
        "service": cat["service_id"], "group": cat["group"],
        "attach": [_pct(x) for x in cat["attach"]], "flags": cat["flags"],
    })))
    free = cat.loc[cat["is_free"], "service_id"].tolist()
    out.append(f"\n**Revenue flag:** {', '.join(free)} are $0 items. Hit-rate metrics count them "
               "like any other service, so they can make a recommender look better without "
               "adding revenue. Later steps report metrics with and without them.\n")

    # --- seasonality
    out.append("## Seasonality (by move month)\n")
    monthly = b.groupby("move_month").size()
    peak = monthly.max()
    out.append("```")
    for m, v in monthly.items():
        out.append(f"{m}  {v:5d}  {'#' * int(round(40 * v / peak))}")
    out.append("```\n")
    by_season = b.groupby("season").size().reindex(["winter", "spring", "summer", "fall"])
    storage_ids = set(services.loc[services["group"] == "storage", "service_id"])
    has_storage = b["booking_id"].isin(items.loc[items["service_id"].isin(storage_ids), "booking_id"])
    st = has_storage.groupby(b["season"]).mean().reindex(by_season.index)
    out.append(_table(pd.DataFrame({
        "season": by_season.index, "bookings": by_season.values,
        "storage attach": [_pct(x) for x in st.values],
    })))
    out.append("")

    # --- rare combinations
    baskets = items.groupby("booking_id")["service_id"].apply(lambda x: tuple(sorted(x)))
    pair_counts = pd.Series([p for bk in baskets for p in combinations(bk, 2)]).value_counts()
    basket_counts = baskets.value_counts()
    out.append("## Rare combinations\n")
    out.append(_table(pd.DataFrame([
        ("distinct service pairs seen", f"{len(pair_counts):,}"),
        ("pairs seen fewer than 5 times", f"{(pair_counts < 5).sum():,} ({_pct((pair_counts < 5).mean())})"),
        ("distinct full baskets", f"{len(basket_counts):,}"),
        ("bookings whose exact basket is unique", _pct((basket_counts == 1).sum() / len(b))),
    ], columns=["metric", "value"])))
    out.append("")

    # --- drift
    b["half"] = b["move_date"].dt.year.astype(str) + "-H" + ((b["move_date"].dt.month > 6) + 1).astype(str)
    fp = b["booking_id"].isin(items.loc[items["service_id"] == "full_packing", "booking_id"])
    out.append("## Drift over time (observable)\n")
    drift = pd.DataFrame({
        "full_packing attach": fp.groupby(b["half"]).mean().map(_pct),
        "labor-only share": (b["move_type"] == "labor_only").groupby(b["half"]).mean().map(_pct),
        "long-distance share": (b["distance_band"] == "long_distance").groupby(b["half"]).mean().map(_pct),
    }).reset_index().rename(columns={"half": "period"})
    out.append(_table(drift))
    out.append("")

    if hidden_segments is not None:
        seg = b[["customer_id", "half"]].merge(hidden_segments, on="customer_id")
        mix = pd.crosstab(seg["half"], seg["segment"], normalize="index").map(_pct)
        out.append("## Hidden segment mix (generator diagnostics only)\n")
        out.append("These labels are hidden from every model. They are shown only to "
                   "confirm the drift built into the generator.\n")
        out.append(_table(mix.reset_index().rename(columns={"half": "period"})))
        out.append("")

    # --- assumptions to check
    out.append("## Assumptions that need attention before publishing\n")
    todo = assumptions.needs_attention()
    if todo:
        out.extend(f"- `{name}`: {reason}" for name, reason in todo)
    else:
        out.append("None.")
    out.append("")
    return "\n".join(out)
