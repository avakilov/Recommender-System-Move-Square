"""Generate the SYNTHETIC dataset and its data-quality report.

    python scripts/generate_data.py            # full dataset, seed 0
    python scripts/generate_data.py --seed 1 --n-bookings 5000
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moverec.assumptions import load_assumptions  # noqa: E402
from moverec.generator import generate, write_csvs  # noqa: E402
from moverec.report import build_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-bookings", type=int, default=None)
    p.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    p.add_argument("--report", default=str(ROOT / "reports" / "data_quality.md"))
    args = p.parse_args()

    a = load_assumptions()
    data = generate(a, seed=args.seed, n_bookings=args.n_bookings)
    write_csvs(data, args.out)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(build_report(data.bookings, data.items, data.services, a, data.hidden_segments))
    print(f"{len(data.bookings):,} bookings, {len(data.items):,} line items -> {args.out}")
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
