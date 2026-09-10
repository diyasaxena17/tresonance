"""Command-line entry point for Phase 4 baseline evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from market_resonance.data import DEFAULT_OUTPUT_PATH
from market_resonance.evaluation import (
    DEFAULT_FIGURE_PATH,
    DEFAULT_RESULTS_PATH,
    run_baseline_evaluation,
    save_baseline_results,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate baseline models and save table/figure outputs."""
    parser = argparse.ArgumentParser(
        description="Evaluate zero-change and linear-regression Treasury baselines."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Cleaned Treasury CSV path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"Metrics CSV path (default: {DEFAULT_RESULTS_PATH}).",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=DEFAULT_FIGURE_PATH,
        help=f"Figure path (default: {DEFAULT_FIGURE_PATH}).",
    )
    args = parser.parse_args(argv)

    run = run_baseline_evaluation(
        data_path=args.data,
        lookback=args.lookback,
        horizon=args.horizon,
    )
    results_path, figure_path = save_baseline_results(
        run,
        results_path=args.results,
        figure_path=args.figure,
    )
    print(run.metrics.to_string(index=False))
    print(f"Saved metrics to {results_path}")
    print(f"Saved figure to {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
