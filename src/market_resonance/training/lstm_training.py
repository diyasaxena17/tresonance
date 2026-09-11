"""Dataset, DataLoader, and training loop helpers for the first LSTM model."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from market_resonance.features import WindowSplits
from market_resonance.models import YieldCurveLSTM

DEFAULT_CHECKPOINT_PATH = Path("reports/models/first_lstm.pt")


@dataclass(frozen=True)
class TrainingConfig:
    """Minimal training settings for the first LSTM."""

    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 50
    patience: int = 5
    seed: int = 42


@dataclass(frozen=True)
class TrainingHistory:
    """Loss history and early-stopping summary."""

    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int
    best_validation_loss: float


class WindowTensorDataset(Dataset):
    """PyTorch Dataset wrapper for supervised window arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_dataloaders(
    splits: WindowSplits,
    batch_size: int = 64,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders from window splits."""
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        WindowTensorDataset(splits.train.X, splits.train.y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        WindowTensorDataset(splits.validation.X, splits.validation.y),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        WindowTensorDataset(splits.test.X, splits.test.y),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, validation_loader, test_loader


def train_lstm_model(
    model: YieldCurveLSTM,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig | None = None,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> TrainingHistory:
    """Train a small LSTM with MSE loss, Adam, and early stopping."""
    if config is None:
        config = TrainingConfig()

    set_deterministic_seed(config.seed)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_losses: list[float] = []
    validation_losses: list[float] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.max_epochs + 1):
        train_loss = _run_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
        validation_loss = _evaluate_loss(model, validation_loader, loss_fn)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "validation_loss": validation_loss,
                    "config": config.__dict__,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    return TrainingHistory(
        train_loss=train_losses,
        validation_loss=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
    )


def _run_epoch(
    model: YieldCurveLSTM,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    weighted_loss = 0.0
    sample_count = 0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.shape[0]
        weighted_loss += loss.item() * batch_size
        sample_count += batch_size
    return weighted_loss / sample_count


def _evaluate_loss(
    model: YieldCurveLSTM,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> float:
    model.eval()
    weighted_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            batch_size = X_batch.shape[0]
            weighted_loss += loss.item() * batch_size
            sample_count += batch_size
    return weighted_loss / sample_count
