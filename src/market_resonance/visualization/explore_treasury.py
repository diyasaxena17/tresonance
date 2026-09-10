"""Command-line entry point for Treasury exploratory analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from market_resonance.data import DEFAULT_OUTPUT_PATH
from market_resonance.visualization.treasury_eda import (
    DEFAULT_FIGURE_DIR,
    generate_treasury_eda_figures,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate Phase 2 Treasury EDA figures."""
    parser = argparse.ArgumentParser(
        description="Generate exploratory Treasury yield figures."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Cleaned Treasury CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help=f"Figure output directory (default: {DEFAULT_FIGURE_DIR}).",
    )
    args = parser.parse_args(argv)

    paths = generate_treasury_eda_figures(args.data, args.output_dir)
    for path in paths:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
