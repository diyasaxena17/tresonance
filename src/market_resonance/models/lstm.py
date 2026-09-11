"""PyTorch LSTM for Treasury yield-change forecasting."""

from __future__ import annotations

import torch
from torch import nn


class YieldCurveLSTM(nn.Module):
    """One-layer LSTM followed by a linear output layer."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 7,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Predict future yield changes from a batch-first sequence tensor."""
        sequence_output, _ = self.lstm(X)
        final_hidden = sequence_output[:, -1, :]
        return self.output_layer(final_hidden)

    def shape_trace(self, X: torch.Tensor) -> dict[str, tuple[int, ...]]:
        """Return tensor shapes at each major model step."""
        with torch.no_grad():
            sequence_output, (hidden_state, cell_state) = self.lstm(X)
            final_hidden = sequence_output[:, -1, :]
            predictions = self.output_layer(final_hidden)
        return {
            "input": tuple(X.shape),
            "lstm_sequence_output": tuple(sequence_output.shape),
            "lstm_hidden_state": tuple(hidden_state.shape),
            "lstm_cell_state": tuple(cell_state.shape),
            "final_hidden": tuple(final_hidden.shape),
            "predictions": tuple(predictions.shape),
        }
