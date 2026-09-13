"""Command-line entry point for stochastic uncertainty inference."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from market_resonance.data import DEFAULT_OUTPUT_PATH
from market_resonance.inference.uncertainty import run_uncertainty_simulation


def main(argv: Sequence[str] | None = None) -> int:
    """Generate Monte Carlo scenarios around the LSTM point forecast."""
    parser = argparse.ArgumentParser(
        description="Generate stochastic uncertainty around an LSTM forecast."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("reports/models/multi_frequency_lstm.pt"),
    )
    parser.add_argument("--scenario-count", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    run = run_uncertainty_simulation(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        scenario_count=args.scenario_count,
        seed=args.seed,
    )
    print(f"Saved summary to {run.summary_path}")
    print(f"Saved scenarios to {run.scenario_path}")
    print(f"Saved ranges to {run.range_path}")
    print(f"Saved fan figure to {run.fan_figure_path}")
    print(f"Saved range figure to {run.range_figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
