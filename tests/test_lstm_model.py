"""Tests for the first PyTorch LSTM model and training loop."""

import numpy as np
import torch

from market_resonance.features import SupervisedWindows, WindowSplits
from market_resonance.models import YieldCurveLSTM
from market_resonance.training import (
    TrainingConfig,
    WindowTensorDataset,
    make_dataloaders,
    train_lstm_model,
)


def test_lstm_forward_and_shape_trace() -> None:
    """The LSTM maps batch-first windows to seven target predictions."""
    model = YieldCurveLSTM(input_size=28, hidden_size=64, output_size=7)
    X = torch.zeros((4, 60, 28), dtype=torch.float32)

    prediction = model(X)
    shapes = model.shape_trace(X)

    assert prediction.shape == (4, 7)
    assert shapes == {
        "input": (4, 60, 28),
        "lstm_sequence_output": (4, 60, 64),
        "lstm_hidden_state": (1, 4, 64),
        "lstm_cell_state": (1, 4, 64),
        "final_hidden": (4, 64),
        "predictions": (4, 7),
    }


def test_window_tensor_dataset_returns_float_tensors() -> None:
    """The Dataset returns one X/y tensor pair by index."""
    dataset = WindowTensorDataset(
        X=np.zeros((3, 5, 2), dtype=float),
        y=np.ones((3, 7), dtype=float),
    )

    X_item, y_item = dataset[0]

    assert len(dataset) == 3
    assert X_item.dtype == torch.float32
    assert y_item.dtype == torch.float32
    assert X_item.shape == (5, 2)
    assert y_item.shape == (7,)


def test_training_loop_saves_best_checkpoint(tmp_path) -> None:
    """A tiny training run records losses and writes the best model."""
    splits = _small_splits()
    train_loader, validation_loader, _ = make_dataloaders(
        splits,
        batch_size=4,
        seed=123,
    )
    model = YieldCurveLSTM(input_size=3, hidden_size=8, output_size=7)
    checkpoint_path = tmp_path / "first_lstm.pt"

    history = train_lstm_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        config=TrainingConfig(
            batch_size=4,
            learning_rate=1e-2,
            max_epochs=3,
            patience=2,
            seed=123,
        ),
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint_path.exists()
    assert len(history.train_loss) >= 1
    assert len(history.validation_loss) == len(history.train_loss)
    assert 1 <= history.best_epoch <= len(history.train_loss)
    assert "model_state_dict" in checkpoint


def _small_splits() -> WindowSplits:
    rng = np.random.default_rng(7)

    def windows(sample_count: int) -> SupervisedWindows:
        return SupervisedWindows(
            X=rng.normal(size=(sample_count, 6, 3)),
            y=rng.normal(size=(sample_count, 7)),
            feature_columns=["a", "b", "c"],
            target_columns=[f"target_{index}" for index in range(7)],
            sample_start_dates=[],
            sample_end_dates=[],
            target_dates=[],
        )

    return WindowSplits(
        train=windows(12),
        validation=windows(8),
        test=windows(8),
    )
