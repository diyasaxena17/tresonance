"""Command-line entry point for downloading Treasury yield data."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .treasury import DEFAULT_OUTPUT_PATH, download_treasury_yields


def main(argv: Sequence[str] | None = None) -> int:
    """Download and save the cleaned Treasury dataset."""
    parser = argparse.ArgumentParser(
        description="Download daily Treasury constant-maturity yields from FRED."
    )
    parser.add_argument("--start-date", help="Inclusive date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Inclusive date in YYYY-MM-DD format.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    args = parser.parse_args(argv)

    cleaned = download_treasury_yields(
        output_path=args.output,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(f"Saved {len(cleaned):,} complete observations to {args.output}")
    start = cleaned["date"].min().date()
    end = cleaned["date"].max().date()
    print(f"Date range: {start} to {end}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
