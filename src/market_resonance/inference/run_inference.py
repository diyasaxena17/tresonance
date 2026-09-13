"""Command-line entry point for LSTM inference."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from market_resonance.data import DEFAULT_OUTPUT_PATH
from market_resonance.inference import (
    DEFAULT_INFERENCE_METRICS_PATH,
    run_lstm_inference,
)
from market_resonance.training import DEFAULT_CHECKPOINT_PATH


def main(argv: Sequence[str] | None = None) -> int:
    """Run saved-model inference and print a seven-maturity forecast."""
    parser = argparse.ArgumentParser(description="Run LSTM Treasury inference.")
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_INFERENCE_METRICS_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args(argv)

    result = run_lstm_inference(
        data_path=args.data,
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        batch_size=args.batch_size,
        repeats=args.repeats,
    )

    print(f"Latest input date: {result.latest_input_date}")
    print(f"Forecast target: {result.forecast_date}")
    print(f"Parameter count: {result.parameter_count:,}")
    print(
        "Latency: "
        f"{result.single_sample_latency_ms:.4f} ms single, "
        f"{result.batch_latency_ms:.4f} ms batch({result.batch_size})"
    )
    print("Forecast:")
    for row in result.forecast:
        print(
            f"  {row['maturity']}: "
            f"{row['predicted_change_bp']:+.4f} bp -> "
            f"{row['forecast_yield_percent']:.4f}%"
        )
    print(f"Saved metrics to {args.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
