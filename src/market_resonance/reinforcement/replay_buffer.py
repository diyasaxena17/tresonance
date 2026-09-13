"""Replay buffer for the Treasury directional DQN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch


class TransitionBatch(NamedTuple):
    """Tensor batch sampled from stored training-period transitions."""

    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


@dataclass(frozen=True)
class Transition:
    """One environment transition for DQN replay."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray | None
    done: bool


class ReplayBuffer:
    """Fixed-capacity replay buffer with seeded random sampling."""

    def __init__(self, capacity: int = 10_000, seed: int = 42) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._storage: list[Transition] = []
        self._position = 0

    def __len__(self) -> int:
        return len(self._storage)

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray | None,
        done: bool,
    ) -> None:
        """Store one transition, overwriting the oldest item at capacity."""
        transition = Transition(
            state=np.asarray(state, dtype=np.float32).copy(),
            action=int(action),
            reward=float(reward),
            next_state=None
            if next_state is None
            else np.asarray(next_state, dtype=np.float32).copy(),
            done=bool(done),
        )
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
        else:
            self._storage[self._position] = transition
        self._position = (self._position + 1) % self.capacity

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a random transition batch as float/long tensors."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if batch_size > len(self):
            raise ValueError("batch_size cannot exceed replay buffer size.")

        indices = self._rng.choice(len(self._storage), size=batch_size, replace=False)
        transitions = [self._storage[index] for index in indices]
        state_shape = transitions[0].state.shape
        next_states = [
            np.zeros(state_shape, dtype=np.float32)
            if transition.next_state is None
            else transition.next_state
            for transition in transitions
        ]
        return TransitionBatch(
            states=torch.as_tensor(
                np.stack([transition.state for transition in transitions]),
                dtype=torch.float32,
            ),
            actions=torch.as_tensor(
                [transition.action for transition in transitions],
                dtype=torch.long,
            ),
            rewards=torch.as_tensor(
                [transition.reward for transition in transitions],
                dtype=torch.float32,
            ),
            next_states=torch.as_tensor(np.stack(next_states), dtype=torch.float32),
            dones=torch.as_tensor(
                [transition.done for transition in transitions],
                dtype=torch.bool,
            ),
        )
