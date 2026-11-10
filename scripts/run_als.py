"""Step 3: implicit ALS with fold-in for partial carts. Writes reports/als.md.

    python scripts/run_als.py                  # validation only
    python scripts/run_als.py --include-test   # unseal the test set (final runs only)
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moverec.als import ImplicitALS  # noqa: E402
from moverec.evaluate import feasibility_mask, top_k  # noqa: E402
from moverec.experiment import (comparison_tables, fmt, prepare, save_params,  # noqa: E402
                                tuned_baselines)
from moverec.report import _table  # noqa: E402

INSPECT = ["full_packing", "small_box", "storage_10x10", "piano", "u_box_5x7", "mattress_bag"]


def diagnostics(ctx, als, global_pop) -> list[str]:
    """Why ALS does or doesn't help: is it a fitting problem or a ranking problem?"""
    c = ctx.carts["val"]
    pop = ctx.X_train.mean(axis=0)
    allowed = feasibility_mask(c.bookings, c.cart, ctx.ds.services)
    small = (c.cart.sum(axis=1) >= 1) & (c.cart.sum(axis=1) <= 2)

    def rec_attach(model):
        ranked = top_k(model.score(c.bookings, c.cart), allowed, 5)[small]
        return pop[ranked[ranked >= 0]].mean()

    Xs = ctx.X_train[:3000]
    top5 = np.argsort(-als.score(None, Xs), axis=1)[:, :5]
    in_sample = np.take_along_axis(Xs, top5, axis=1).mean()
    return [
        "## Diagnostics\n",
        _table(pd.DataFrame([
            ("in-sample precision@5 (full training baskets folded in)", f"{in_sample:.3f}"),
            ("mean training attach rate of top-5 recs, 1-2 item carts: ALS", f"{rec_attach(als):.3f}"),
            ("same, global popularity", f"{rec_attach(global_pop):.3f}"),
            ("mean training attach rate of the hidden targets, 1-2 item carts",
             f"{(c.target[small] * pop).sum() / c.target[small].sum():.3f}"),
        ], columns=["check", "value"])),
        "",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--out", default=str(ROOT / "reports" / "als.md"))
    args = p.parse_args()

    ctx = prepare(include_test=args.include_test)
    g = ctx.cfg["als"]
    train, X = ctx.split.train, ctx.X_train

    rows = []
    for f, reg, alpha in itertools.product(g["factors"], g["reg"], g["alpha"]):
        m = ImplicitALS(f, reg, alpha, g["iterations"], g["seed"]).fit(train, X)
        rows.append({"factors": f, "reg": reg, "alpha": alpha, ctx.key: ctx.score(m)})
    grid = pd.DataFrame(rows).sort_values(ctx.key, ascending=False).reset_index(drop=True)
    best = grid.iloc[0]
    params = dict(factors=int(best["factors"]), reg=float(best["reg"]), alpha=float(best["alpha"]),
                  iterations=g["iterations"])

    save_params("als", {**params, "seed": g["seed"]})

    seeds = [ctx.score(ImplicitALS(**params, seed=s).fit(train, X)) for s in g["stability_seeds"]]
    als = ImplicitALS(**params, seed=g["seed"]).fit(train, X)
    baselines, _ = tuned_baselines(ctx)
    models = baselines + [als]

    sids = ctx.ds.service_ids
    neigh = pd.DataFrame({
        "service": INSPECT,
        "nearest by item-factor cosine": [
            ", ".join(f"{sids[j]} ({s:.2f})" for j, s in als.similar_items(sids.index(sid), 4))
            for sid in INSPECT],
    })

    out = ["# Step 3: implicit ALS + fold-in (SYNTHETIC data)\n",
           "> All data is synthetic. These numbers validate the pipeline, not real-world performance.\n",
           "Rows of the training matrix are bookings (almost every customer is new), columns are "
           "the 45 services. At evaluation time each partial cart is folded in with the item "
           "factors fixed (a ridge least-squares solve), then services are ranked by x . y.\n",
           f"## Grid search (validation {ctx.key}, {len(grid)} configs, {g['iterations']} iterations)\n",
           "Top 8:\n", _table(fmt(grid.head(8))), "",
           f"Worst config: {grid[ctx.key].min():.3f}. Chosen: factors={params['factors']}, "
           f"reg={params['reg']}, alpha={params['alpha']}.\n",
           f"Re-fit with init seeds {g['stability_seeds']}: {ctx.key} = "
           + ", ".join(f"{s:.4f}" for s in seeds) + f" (spread {max(seeds) - min(seeds):.4f}).\n",
           "Caveat: ALS had a 27-point grid on validation, the baselines had at most 4, so "
           "ALS's validation numbers are slightly optimistic. The sealed test set settles it.\n"]
    for part in ctx.carts:
        out += comparison_tables(ctx, models, part)
    out += ["## What the item factors learned\n", _table(neigh), ""]
    out += diagnostics(ctx, als, baselines[0])
    if not args.include_test:
        out.append("_Test set sealed. It is evaluated once, in step 6._\n")
    Path(args.out).write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
