"""Step 5: weighted hybrid tuned on validation. Writes reports/hybrid.md.

    python scripts/run_hybrid.py                  # validation only
    python scripts/run_hybrid.py --include-test   # unseal the test set (final runs only)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moverec.baselines import CoOccurrenceLift, ProfilePopularity  # noqa: E402
from moverec.content import ContentKNN  # noqa: E402
from moverec.evaluate import (CART_SIZE_BINS, CART_SIZE_LABELS, EvalCarts, evaluate,  # noqa: E402
                              paired_bootstrap, summarize)
from moverec.experiment import (comparison_tables, fmt, load_params, prepare,  # noqa: E402
                                save_params, tuned_als, tuned_baselines)
from moverec.hybrid import FixedScores, Hybrid, combine, component_scores, tune_weights  # noqa: E402
from moverec.report import _table  # noqa: E402


def subset(c: EvalCarts, idx: np.ndarray) -> EvalCarts:
    return EvalCarts(c.bookings.iloc[idx].reset_index(drop=True), c.cart[idx], c.target[idx])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--out", default=str(ROOT / "reports" / "hybrid.md"))
    args = p.parse_args()

    ctx = prepare(include_test=args.include_test)
    hc = ctx.cfg["hybrid"]
    train, X = ctx.split.train, ctx.X_train
    ev = ctx.cfg["eval"]

    components = {
        "profile": ProfilePopularity(load_params("baselines")["profile_smoothing"]).fit(train, X),
        "content": ContentKNN(**load_params("content")).fit(train, X),
        "cooc_lift": CoOccurrenceLift(hc["cooc_lift_prior"]).fit(train, X),
        "als": tuned_als(ctx),
    }
    names = list(components)
    val = ctx.carts["val"]
    scores = component_scores(components, val.bookings, val.cart)

    def metric_on(carts):
        return lambda s: float(summarize(evaluate(FixedScores(s), carts, ctx.ds.services,
                                                  ev["ks"], ev["map_k"]))[ctx.key])

    def sub_scores(idx):
        return {n: s[idx] for n, s in scores.items()}

    # 1. tune on all validation carts
    weights, best_val, table = tune_weights(scores, metric_on(val), hc["weight_grid"])
    save_params("hybrid", {"weights": weights, "cooc_lift_prior": hc["cooc_lift_prior"]})

    # 2. honest estimate: tune on one half of validation, score on the other, and swap
    rng = np.random.default_rng(hc["crossfit_seed"])
    perm = rng.permutation(len(val.cart))
    halves = [np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:])]
    crossfit = []
    for a, b in ((0, 1), (1, 0)):
        w, _, _ = tune_weights(sub_scores(halves[a]), metric_on(subset(val, halves[a])), hc["weight_grid"])
        held = metric_on(subset(val, halves[b]))(combine(sub_scores(halves[b]), w))
        base = metric_on(subset(val, halves[b]))(scores["profile"][halves[b]])
        crossfit.append({"tuned on half": a + 1, "scored on half": b + 1,
                         **{f"w_{n}": w[n] for n in names},
                         f"hybrid {ctx.key}": held, f"profile_pop {ctx.key}": base})
    crossfit = pd.DataFrame(crossfit)

    # 3. ablations: drop one component, re-tune the rest
    abl = [{"variant": "full hybrid", **{f"w_{n}": weights[n] for n in names}, ctx.key: best_val}]
    for n in names:
        w, m, _ = tune_weights(scores, metric_on(val), hc["weight_grid"], fixed_zero=(n,))
        abl.append({"variant": f"without {n}", **{f"w_{k}": w[k] for k in names}, ctx.key: m})
    for keep in (["profile", "cooc_lift"], ["profile", "content"]):
        drop = tuple(n for n in names if n not in keep)
        w, m, _ = tune_weights(scores, metric_on(val), hc["weight_grid"], fixed_zero=drop)
        abl.append({"variant": " + ".join(keep) + " only", **{f"w_{k}": w[k] for k in names}, ctx.key: m})
    abl = pd.DataFrame(abl)

    # 4. compare with everything so far
    hybrid = Hybrid(components, weights)
    baselines, _ = tuned_baselines(ctx)
    models = baselines + [components["als"], components["content"], components["cooc_lift"], hybrid]

    sig = []
    for part in ctx.carts:
        h = ctx.run(hybrid, part)
        b = ctx.run(baselines[1], part)
        bucket = pd.cut(h["cart_size"], CART_SIZE_BINS, labels=CART_SIZE_LABELS)
        for label in ["all"] + CART_SIZE_LABELS:
            m = np.ones(len(h), bool) if label == "all" else (bucket == label).to_numpy()
            r = paired_bootstrap(h[ctx.key].to_numpy()[m], b[ctx.key].to_numpy()[m])
            sig.append((part, label, int(m.sum()), f"{r['diff']:+.4f}",
                        f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"))

    top = table.sort_values("metric", ascending=False).head(8).rename(columns={"metric": ctx.key})
    out = ["# Step 5: weighted hybrid (SYNTHETIC data)\n",
           "> All data is synthetic. These numbers validate the pipeline, not real-world performance.\n",
           "hybrid(j) = sum over components of weight x row-standardised score. Profile and content "
           "scores are logged first (probabilities multiply, so logs add). The co-occurrence "
           "component here is the **lift** version, mean log P(j|i)/P(j) over cart items, so it "
           "carries the cart signal without counting popularity twice. Cart components are flat "
           "on empty carts and abstain there.\n",
           f"## Weight search ({len(table)} combinations, validation {ctx.key})\n",
           "Top 8:\n", _table(fmt(top)), "",
           "Chosen weights: " + ", ".join(f"{n}={weights[n]}" for n in names)
           + f" -> validation {ctx.key} {best_val:.4f}.\n",
           "## Honest estimate: 2-fold cross-fit inside validation\n",
           "Weights tuned on one half of the validation carts, scored on the other half.\n",
           _table(fmt(crossfit)), "",
           "## Ablations (validation, remaining weights re-tuned)\n",
           _table(fmt(abl)), ""]
    for part in ctx.carts:
        out += comparison_tables(ctx, models, part)
    out += [f"## Hybrid minus profile popularity (paired bootstrap, {ctx.key})\n",
            _table(pd.DataFrame(sig, columns=["part", "cart size", "carts", "difference", "95% interval"])),
            "\nAn interval that excludes 0 is a detectable difference.\n"]
    if not args.include_test:
        out.append("_Test set sealed. It is evaluated once, in step 6._\n")
    Path(args.out).write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
