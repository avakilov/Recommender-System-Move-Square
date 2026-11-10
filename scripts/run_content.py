"""Step 4: move-profile cosine kNN for empty carts. Writes reports/content.md.

    python scripts/run_content.py                  # validation only
    python scripts/run_content.py --include-test   # unseal the test set (final runs only)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moverec.content import ContentKNN  # noqa: E402
from moverec.evaluate import (EvalCarts, feasibility_mask, mean_rank_when_hidden,  # noqa: E402
                              paired_bootstrap, top_k)

# Services whose purchase the generator ties to an observed profile feature.
NICHE = ["stairs_protector", "day_of_packing", "u_box_5x7", "storage_10x10",
         "full_packing", "mattress_bag", "piano"]
from moverec.experiment import (comparison_tables, fmt, prepare, save_params,  # noqa: E402
                                tuned_als, tuned_baselines)
from moverec.report import _table  # noqa: E402


def empty_only(c: EvalCarts) -> EvalCarts:
    rows = c.cart.sum(axis=1) == 0
    return EvalCarts(c.bookings[rows].reset_index(drop=True), c.cart[rows], c.target[rows])


def per_service_recall(ctx, model, carts: EvalCarts, k: int = 5) -> pd.Series:
    """For each service: of the empty carts where it is hidden, share with it in the top k."""
    allowed = feasibility_mask(carts.bookings, carts.cart, ctx.ds.services)
    ranked = top_k(model.score(carts.bookings, carts.cart), allowed, k)
    in_top = np.zeros_like(carts.target)
    rows = np.repeat(np.arange(len(ranked)), k)
    ok = ranked.ravel() >= 0
    in_top[rows[ok], ranked.ravel()[ok]] = True
    n = carts.target.sum(axis=0)
    rate = (in_top & carts.target).sum(axis=0) / np.maximum(n, 1)
    return pd.Series(rate, index=ctx.ds.service_ids).where(n >= 20)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--out", default=str(ROOT / "reports" / "content.md"))
    args = p.parse_args()

    ctx = prepare(include_test=args.include_test)
    g = ctx.cfg["content"]
    train, X = ctx.split.train, ctx.X_train
    full_val = ctx.carts["val"]
    ctx.carts["val_empty"] = empty_only(full_val)

    rows = []
    for wname, weights in g["group_weightings"].items():
        for k in g["k"]:
            m = ContentKNN(k, weights).fit(train, X)
            rows.append({"weighting": wname, "k": k,
                         f"{ctx.key} empty carts": ctx.score(m, "val_empty"),
                         f"{ctx.key} all carts": ctx.score(m, "val")})
    grid = pd.DataFrame(rows)
    best = grid.sort_values(f"{ctx.key} empty carts", ascending=False).iloc[0]
    params = {"k": int(best["k"]), "group_weights": g["group_weightings"][best["weighting"]]}
    save_params("content", params)

    content = ContentKNN(**params).fit(train, X)
    baselines, _ = tuned_baselines(ctx)
    profile_pop = baselines[1]
    models = baselines + [tuned_als(ctx), content]

    rec = pd.DataFrame({
        "profile_popularity": per_service_recall(ctx, profile_pop, ctx.carts["val_empty"]),
        "content_knn": per_service_recall(ctx, content, ctx.carts["val_empty"]),
    }).dropna()
    rec["difference"] = rec["content_knn"] - rec["profile_popularity"]
    rec = rec.sort_values("difference", ascending=False)

    out = ["# Step 4: move-profile cosine kNN (SYNTHETIC data)\n",
           "> All data is synthetic. These numbers validate the pipeline, not real-world performance.\n",
           "Each booking's move profile (home size, labor-only, log distance, log lead time, "
           "origin/destination access, move month as sin/cos) is standardised, balanced so every "
           "feature group carries equal variance, and compared by cosine similarity. A service's "
           "score is the similarity-weighted share of the k nearest training bookings that bought "
           "it. The cart is ignored, so this model is aimed at empty carts.\n",
           f"## Tuning (validation; selected on empty carts, n={len(ctx.carts['val_empty'].cart):,})\n",
           _table(fmt(grid)), "",
           f"Chosen: k={params['k']}, weighting={best['weighting']}.\n"]
    for part in [x for x in ctx.carts if x != "val_empty"]:
        out += comparison_tables(ctx, models, part)
    sig = []
    for label, part in (("all validation carts", "val"), ("empty validation carts", "val_empty")):
        r = paired_bootstrap(ctx.run(content, part)[ctx.key], ctx.run(profile_pop, part)[ctx.key])
        sig.append((label, f"{r['diff']:+.4f}", f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"))
    out += [f"## Is content kNN better than profile popularity? (paired bootstrap, {ctx.key})\n",
            _table(pd.DataFrame(sig, columns=["carts", "kNN minus profile pop", "95% interval"])),
            "\nAn interval that contains 0 means no detectable difference.\n"]

    ranks = pd.DataFrame({
        "profile_popularity": mean_rank_when_hidden(profile_pop, full_val, ctx.ds.services),
        "content_knn": mean_rank_when_hidden(content, full_val, ctx.ds.services),
    }).loc[NICHE]
    out += ["## Niche services: mean rank when hidden (all validation carts, 1 = top)\n",
            "Low-attach services rarely reach the top 5 under any model, so MAP can't show "
            "whether profile signals (stairs, lead time, season) were learned. Rank can.\n",
            _table(fmt(ranks.reset_index(names="service"))), ""]

    out += ["## Empty carts: per-service recall@5, content kNN vs profile popularity\n",
            "Share of empty validation carts in which the service was hidden AND ranked in the "
            "top 5. Services hidden in fewer than 20 empty carts are left out.\n",
            "Largest gains:\n", _table(fmt(rec.head(8).reset_index(names="service"))), "",
            "Largest losses:\n", _table(fmt(rec.tail(5).iloc[::-1].reset_index(names="service"))), ""]
    if not args.include_test:
        out.append("_Test set sealed. It is evaluated once, in step 6._\n")
    Path(args.out).write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
