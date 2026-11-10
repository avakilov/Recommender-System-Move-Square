"""Step 2: time split + baselines. Writes reports/baselines.md.

    python scripts/run_baselines.py                 # validation only
    python scripts/run_baselines.py --include-test  # unseal the test set (final runs only)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moverec.experiment import comparison_tables, prepare, tuned_baselines  # noqa: E402
from moverec.report import _table  # noqa: E402
from moverec.split import split_summary  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-test", action="store_true")
    p.add_argument("--out", default=str(ROOT / "reports" / "baselines.md"))
    args = p.parse_args()

    ctx = prepare(include_test=args.include_test)
    models, tuning = tuned_baselines(ctx)
    best = max(tuning, key=tuning.get)

    out = ["# Step 2: time split and baselines (SYNTHETIC data)\n",
           "> All data is synthetic. These numbers validate the pipeline, not real-world performance.\n",
           "## Split (by booking time)\n", _table(split_summary(ctx.split)), "",
           f"Train covers {len(ctx.X_train):,} baskets. Evaluated carts: "
           + ", ".join(f"{n} {len(c.cart):,}" for n, c in ctx.carts.items()) + ".\n",
           "## Tuning\n",
           f"Profile-popularity smoothing m, chosen by validation {ctx.key}: "
           + ", ".join(f"m={m}: {v:.4f}" for m, v in tuning.items()) + f" -> **m={best}**.\n"]
    for part in ctx.carts:
        out += comparison_tables(ctx, models, part)
    if not args.include_test:
        out.append("_Test set sealed. It is evaluated once, in step 6._\n")
    Path(args.out).write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
