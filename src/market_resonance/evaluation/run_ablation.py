"""Command-line entry point for the multi-frequency ablation study."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from market_resonance.data import DEFAULT_OUTPUT_PATH
from market_resonance.evaluation.ablation import run_ablation_study
from market_resonance.training import TrainingConfig


def main(argv: Sequence[str] | None = None) -> int:
    """Run the daily-vs-multi-frequency LSTM ablation study."""
    parser = argparse.ArgumentParser(description="Run LSTM feature ablation study.")
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    run = run_ablation_study(
        data_path=args.data,
        config=TrainingConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
        ),
    )
    overall_test = run.metrics[
        (run.metrics["split"] == "test") & (run.metrics["maturity"] == "overall")
    ]
    print(overall_test.to_string(index=False))
    print(f"Hypothesis conclusion: {run.summary['conclusion']}")
    print(f"Saved table to {run.table_path}")
    print(f"Saved figure to {run.figure_path}")
    print(f"Saved loss figure to {run.loss_figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
