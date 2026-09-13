"""Small Deep Q-Network for the Treasury directional environment."""

from __future__ import annotations

import torch
from torch import nn

from market_resonance.reinforcement.environment import (
    ACTION_FALL,
    ACTION_RISE,
    VALID_ACTIONS,
)


class TreasuryDQN(nn.Module):
    """Tiny feed-forward Q-network for three 10Y direction actions.

    The model maps one environment state vector to three Q-values, ordered as
    ``0 = FALL``, ``1 = FLAT``, and ``2 = RISE``. It intentionally contains no
    training loop; this module only defines inference-time Q-value computation
    and simple epsilon-greedy action selection for the educational RL extension.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = len(VALID_ACTIONS),
        seed: int | None = None,
    ) -> None:
        if input_dim < 1:
            raise ValueError("input_dim must be at least 1.")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1.")
        if output_dim != len(VALID_ACTIONS):
            raise ValueError("output_dim must equal 3 for FALL, FLAT, and RISE.")
        if seed is not None:
            torch.manual_seed(seed)

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self._generator = torch.Generator().manual_seed(seed or 0)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return Q-values for a batch of state vectors."""
        self._validate_batch(X)
        return self.network(X)

    def greedy_action(self, state: torch.Tensor) -> int:
        """Select the highest-Q action for one unbatched state."""
        state_batch = self._state_batch(state)
        with torch.no_grad():
            q_values = self(state_batch)
        return int(torch.argmax(q_values, dim=1).item())

    def epsilon_greedy_action(self, state: torch.Tensor, epsilon: float) -> int:
        """Select an action using epsilon-greedy exploration.

        ``epsilon=0`` always returns the greedy action. ``epsilon=1`` always
        samples uniformly from the three valid actions using the model's seeded
        PyTorch generator.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")

        explore = torch.rand((), generator=self._generator).item() < epsilon
        if explore:
            return int(
                torch.randint(
                    low=ACTION_FALL,
                    high=ACTION_RISE + 1,
                    size=(),
                    generator=self._generator,
                ).item()
            )
        return self.greedy_action(state)

    def shape_trace(self, X: torch.Tensor) -> dict[str, tuple[int, ...]]:
        """Return tensor shapes at each linear block for a batch of states."""
        self._validate_batch(X)
        with torch.no_grad():
            hidden_1 = self.network[0](X)
            activated_1 = self.network[1](hidden_1)
            hidden_2 = self.network[2](activated_1)
            activated_2 = self.network[3](hidden_2)
            q_values = self.network[4](activated_2)
        return {
            "input": tuple(X.shape),
            "hidden_1": tuple(hidden_1.shape),
            "activated_1": tuple(activated_1.shape),
            "hidden_2": tuple(hidden_2.shape),
            "activated_2": tuple(activated_2.shape),
            "q_values": tuple(q_values.shape),
        }

    def _state_batch(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 1:
            raise ValueError("state must be a 1D tensor.")
        if state.shape[0] != self.input_dim:
            raise ValueError(f"state must have shape ({self.input_dim},).")
        return state.to(dtype=torch.float32).unsqueeze(0)

    def _validate_batch(self, X: torch.Tensor) -> None:
        if X.ndim != 2:
            raise ValueError("X must be a 2D tensor with shape (batch, input_dim).")
        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"X must have shape (batch, {self.input_dim}); got {tuple(X.shape)}."
            )
