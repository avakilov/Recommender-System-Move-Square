"""Step 6: the one-time evaluation on the sealed test period. Writes reports/final_eval.md.

Nothing is tuned here. Every hyperparameter and hybrid weight comes from
reports/params/*.json, chosen on validation in steps 2-5. Models are refit on
train + val (more recent data) and scored on test carts drawn from a fixed seed.

Each run is appended to reports/test_runs.log so repeated peeking is visible.

    python scripts/run_final_eval.py
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moverec.als import ImplicitALS  # noqa: E402
from moverec.baselines import (CoOccurrence, CoOccurrenceLift, GlobalPopularity,  # noqa: E402
                               ProfilePopularity)
from moverec.content import ContentKNN  # noqa: E402
from moverec.data import basket_matrix  # noqa: E402
from moverec.evaluate import (CART_SIZE_BINS, CART_SIZE_LABELS, by_cart_size, by_label,  # noqa: E402
                              evaluate, head_services, paired_bootstrap, summarize)
from moverec.experiment import PARAMS_DIR, fmt, load_params, prepare  # noqa: E402
from moverec.generator import SEASON_OF_MONTH  # noqa: E402  (public calendar mapping only)
from moverec.hybrid import FixedScores, Hybrid, combine, component_scores, tune_weights  # noqa: E402
from moverec.report import _table  # noqa: E402

LOG = ROOT / "reports" / "test_runs.log"
ORDER = ["global_popularity", "profile_popularity", "co_occurrence", "implicit_als",
         "content_knn", "cooc_lift", "hybrid"]


def build_components(bookings, X):
    """Hybrid components with the hyperparameters chosen on validation."""
    hp = load_params("hybrid")
    return {
        "profile": ProfilePopularity(load_params("baselines")["profile_smoothing"]).fit(bookings, X),
        "content": ContentKNN(**load_params("content")).fit(bookings, X),
        "cooc_lift": CoOccurrenceLift(hp["cooc_lift_prior"]).fit(bookings, X),
        "als": ImplicitALS(**load_params("als")).fit(bookings, X),
    }


def build_models(bookings, X, components):
    return {
        "global_popularity": GlobalPopularity().fit(bookings, X),
        "profile_popularity": components["profile"],
        "co_occurrence": CoOccurrence().fit(bookings, X),
        "implicit_als": components["als"],
        "content_knn": components["content"],
        "cooc_lift": components["cooc_lift"],
        "hybrid": Hybrid(components, load_params("hybrid")["weights"]),
    }


def ablation_weights(ctx, components_train) -> dict[str, dict]:
    """Re-derive the step-5 ablation weights on VALIDATION (train-fit components)."""
    val = ctx.carts["val"]
    scores = component_scores(components_train, val.bookings, val.cart)
    ev = ctx.cfg["eval"]

    def metric(s):
        return float(summarize(evaluate(FixedScores(s), val, ctx.ds.services, ev["ks"], ev["map_k"]))[ctx.key])

    grid = ctx.cfg["hybrid"]["weight_grid"]
    names = list(scores)
    variants = {"full hybrid": load_params("hybrid")["weights"]}
    for n in names:
        variants[f"without {n}"] = tune_weights(scores, metric, grid, fixed_zero=(n,))[0]
    for keep in (["profile", "content"], ["profile", "cooc_lift"]):
        drop = tuple(n for n in names if n not in keep)
        variants[" + ".join(keep) + " only"] = tune_weights(scores, metric, grid, fixed_zero=drop)[0]
    return variants


def log_run(summary: dict) -> int:
    params = "".join(p.read_text() for p in sorted(PARAMS_DIR.glob("*.json")))
    entry = {"time": datetime.now().isoformat(timespec="seconds"),
             "params_sha256": hashlib.sha256(params.encode()).hexdigest()[:12], **summary}
    previous = LOG.read_text().count("\n") if LOG.exists() else 0
    with LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return previous


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "reports" / "final_eval.md"))
    p.add_argument("--dry-run-on-val", action="store_true",
                   help="exercise the whole script with validation carts standing in for test; "
                        "the test set stays sealed and nothing is logged")
    args = p.parse_args()

    if args.dry_run_on_val:
        ctx = prepare(include_test=False)
        ctx.carts["test"] = ctx.carts["val"]
    else:
        ctx = prepare(include_test=True)
    ev, fin = ctx.cfg["eval"], ctx.cfg["final"]
    services, key = ctx.ds.services, ctx.key
    test = ctx.carts["test"]

    # models fit on train only: for validation numbers and ablation weights
    comp_train = build_components(ctx.split.train, ctx.X_train)
    models_train = build_models(ctx.split.train, ctx.X_train, comp_train)
    variants = ablation_weights(ctx, comp_train)

    # final models: refit on train + val
    if fin["refit_on_train_plus_val"]:
        fit_b = pd.concat([ctx.split.train, ctx.split.val], ignore_index=True)
    else:
        fit_b = ctx.split.train
    X_fit = basket_matrix(fit_b, ctx.ds.items, ctx.ds.service_ids)
    comp = build_components(fit_b, X_fit)
    models = build_models(fit_b, X_fit, comp)

    # score every model once on the test carts
    S = {name: models[name].score(test.bookings, test.cart) for name in ORDER}
    head = head_services(X_fit, services, fin["head_min_attach"])
    views = {
        "all services": None,
        "excluding $0 items (" + ", ".join(ctx.free) + ")": ctx.free,
        f"long tail: excluding services in >= {fin['head_min_attach']:.0%} of bookings": head,
    }
    per = {v: {n: evaluate(FixedScores(S[n]), test, services, ev["ks"], ev["map_k"], exclude=ex)
               for n in ORDER} for v, ex in views.items()}

    out = ["# Step 6: final evaluation on the sealed test period (SYNTHETIC data)\n",
           "> All data is synthetic. These numbers validate the pipeline, not real-world performance. "
           "Offline evaluation only; no A/B test.\n",
           "## Protocol\n",
           f"- Test = bookings made {test.bookings['booking_ts'].min():%Y-%m-%d} to "
           f"{test.bookings['booking_ts'].max():%Y-%m-%d} (summer + fall), after every training "
           "and validation booking. Most moves fall in seasons the validation period (spring) never saw.",
           f"- {len(test.cart):,} partial carts: a random subset of each booking's services is "
           "visible (possibly none), the rest are hidden targets. Same seeded procedure as validation.",
           "- Services already in the cart, and services the catalog rules forbid (a second truck, a "
           "truck on a labor-only move, a second storage unit), are never recommended.",
           f"- Models refit on train + val ({len(fit_b):,} bookings) with hyperparameters and hybrid "
           "weights chosen on validation. **Nothing was tuned on test.**",
           f"- Long-tail view drops the {len(head)} services bought in at least "
           f"{fin['head_min_attach']:.0%} of bookings: " + ", ".join(head) + ".", ""]

    for v in views:
        tbl = pd.DataFrame({n: summarize(per[v][n]) for n in ORDER}).T
        out += [f"## Test: {v}\n", _table(fmt(tbl.reset_index(names="model"))), ""]

    sliced = {n: by_cart_size(per["all services"][n]) for n in ORDER}
    tbl = pd.DataFrame({n: s[key] for n, s in sliced.items()})
    tbl.insert(0, "carts", sliced["hybrid"]["carts"])
    out += [f"## Test by visible cart size ({key}, all services)\n",
            _table(fmt(tbl.reset_index(names="cart size"))), ""]

    season = test.bookings["move_date"].dt.month.map(SEASON_OF_MONTH)
    sl = {n: by_label(per["all services"][n], season) for n in ORDER}
    tbl = pd.DataFrame({n: s[key] for n, s in sl.items()})
    tbl = tbl.reindex([s for s in ["spring", "summer", "fall", "winter"] if s in tbl.index])
    tbl.insert(0, "carts", sl["hybrid"]["carts"])
    out += [f"## Test by move season ({key}, all services)\n",
            _table(fmt(tbl.reset_index(names="move season"))), ""]

    # validation (train-fit) vs test (train+val-fit)
    vt = []
    for n in ORDER:
        v = summarize(evaluate(models_train[n], ctx.carts["val"], services, ev["ks"], ev["map_k"]))[key]
        t = summarize(per["all services"][n])[key]
        vt.append((n, f"{v:.3f}", f"{t:.3f}", f"{t - v:+.3f}"))
    out += [f"## Validation vs test ({key}, all services)\n",
            _table(pd.DataFrame(vt, columns=["model", "validation", "test", "change"])), ""]

    # significance
    pairs = [("hybrid", "profile_popularity"), ("hybrid", "global_popularity"),
             ("profile_popularity", "global_popularity"), ("content_knn", "profile_popularity"),
             ("implicit_als", "global_popularity"), ("implicit_als", "co_occurrence")]
    sig = []
    for v in list(views)[::2]:   # all services + long tail
        for a, b in pairs:
            r = paired_bootstrap(per[v][a][key], per[v][b][key])
            sig.append((v.split(":")[0], f"{a} - {b}", f"{r['diff']:+.4f}",
                        f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
                        "yes" if r["ci_low"] > 0 or r["ci_high"] < 0 else "no"))
    h, b = per["all services"]["hybrid"], per["all services"]["profile_popularity"]
    bucket = pd.cut(h["cart_size"], CART_SIZE_BINS, labels=CART_SIZE_LABELS).to_numpy()
    for lab in CART_SIZE_LABELS:
        m = bucket == lab
        r = paired_bootstrap(h[key].to_numpy()[m], b[key].to_numpy()[m])
        sig.append((f"cart size {lab}", "hybrid - profile_popularity", f"{r['diff']:+.4f}",
                    f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
                    "yes" if r["ci_low"] > 0 or r["ci_high"] < 0 else "no"))
    out += [f"## Paired bootstrap on test ({key}, 2,000 resamples of carts)\n",
            _table(pd.DataFrame(sig, columns=["view", "comparison", "difference", "95% interval",
                                              "detectable"])), ""]

    # hybrid ablations: weights from validation, scored on test
    cs = component_scores(comp, test.bookings, test.cart)
    abl = []
    for name, w in variants.items():
        m_all = summarize(evaluate(FixedScores(combine(cs, w)), test, services, ev["ks"], ev["map_k"]))[key]
        m_tail = summarize(evaluate(FixedScores(combine(cs, w)), test, services, ev["ks"], ev["map_k"],
                                    exclude=head))[key]
        abl.append({"variant": name, **{f"w_{k}": w[k] for k in cs}, f"{key} all": m_all,
                    f"{key} long tail": m_tail})
    out += ["## Hybrid ablations on test (weights chosen on validation)\n",
            _table(fmt(pd.DataFrame(abl))), ""]

    if args.dry_run_on_val:
        Path(args.out).write_text("\n".join(out))
        print(f"dry run on validation carts -> {args.out} (test still sealed, nothing logged)")
        return
    n_prev = log_run({n: round(float(summarize(per["all services"][n])[key]), 4) for n in ORDER})
    out.append(f"_Test run logged to reports/test_runs.log (previous runs: {n_prev})._\n")
    Path(args.out).write_text("\n".join(out))
    print("\n".join(out))
    if n_prev:
        print(f"\nWARNING: the test set had already been evaluated {n_prev} time(s).")


if __name__ == "__main__":
    main()
