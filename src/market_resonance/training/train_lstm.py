"""Command-line entry point for training the first small LSTM."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from market_resonance.data import DEFAULT_OUTPUT_PATH
from market_resonance.features import (
    build_supervised_windows,
    chronological_train_validation_test_split,
    standardize_splits,
)
from market_resonance.models import YieldCurveLSTM
from market_resonance.training import (
    DEFAULT_CHECKPOINT_PATH,
    TrainingConfig,
    make_dataloaders,
    set_deterministic_seed,
    train_lstm_model,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Train the first intentionally small LSTM model."""
    parser = argparse.ArgumentParser(description="Train a small Treasury LSTM.")
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    set_deterministic_seed(args.seed)
    yields = pd.read_csv(args.data, parse_dates=["date"])
    windows = build_supervised_windows(
        yields,
        lookback=args.lookback,
        horizon=args.horizon,
    )
    splits = chronological_train_validation_test_split(windows)
    standardized_splits, _ = standardize_splits(splits)
    train_loader, validation_loader, _ = make_dataloaders(
        standardized_splits,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    model = YieldCurveLSTM(
        input_size=standardized_splits.train.X.shape[-1],
        hidden_size=args.hidden_size,
        output_size=standardized_splits.train.y.shape[-1],
    )
    first_batch_X, _ = next(iter(train_loader))
    print("Tensor shapes:")
    for name, shape in model.shape_trace(first_batch_X).items():
        print(f"  {name}: {shape}")

    config = TrainingConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )
    history = train_lstm_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=config,
        checkpoint_path=args.checkpoint,
    )
    print(f"Best epoch: {history.best_epoch}")
    print(f"Best validation MSE: {history.best_validation_loss:.8f}")
    print(f"Saved checkpoint to {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
