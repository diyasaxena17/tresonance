"""Tests for DQN replay-buffer behavior."""

import numpy as np
import pytest

from market_resonance.reinforcement import ReplayBuffer


def test_replay_buffer_appends_and_samples_tensor_batch() -> None:
    buffer = ReplayBuffer(capacity=4, seed=123)
    for index in range(4):
        state = np.full(5, index, dtype=float)
        buffer.append(
            state=state,
            action=index % 3,
            reward=float(index),
            next_state=state + 1,
            done=False,
        )

    batch = buffer.sample(batch_size=3)

    assert len(buffer) == 4
    assert batch.states.shape == (3, 5)
    assert batch.actions.shape == (3,)
    assert batch.rewards.shape == (3,)
    assert batch.next_states.shape == (3, 5)
    assert batch.dones.shape == (3,)


def test_replay_buffer_overwrites_oldest_transition_at_capacity() -> None:
    buffer = ReplayBuffer(capacity=2, seed=123)
    buffer.append(np.zeros(5), 0, 0.0, np.ones(5), False)
    buffer.append(np.ones(5), 1, 1.0, np.ones(5) * 2, False)
    buffer.append(np.ones(5) * 2, 2, 2.0, None, True)

    batch = buffer.sample(batch_size=2)

    assert len(buffer) == 2
    assert set(batch.rewards.tolist()) == {1.0, 2.0}
    terminal_index = batch.dones.tolist().index(True)
    assert batch.next_states[terminal_index].tolist() == [0.0] * 5


def test_replay_buffer_seeded_sampling_is_reproducible() -> None:
    first = ReplayBuffer(capacity=6, seed=77)
    second = ReplayBuffer(capacity=6, seed=77)
    for index in range(6):
        for buffer in (first, second):
            buffer.append(np.full(5, index), index % 3, float(index), None, True)

    first_batch = first.sample(batch_size=4)
    second_batch = second.sample(batch_size=4)

    assert first_batch.rewards.tolist() == second_batch.rewards.tolist()


def test_replay_buffer_rejects_invalid_sizes() -> None:
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=0)

    buffer = ReplayBuffer(capacity=2)
    buffer.append(np.zeros(5), 0, 0.0, None, True)
    with pytest.raises(ValueError, match="batch_size"):
        buffer.sample(batch_size=0)
    with pytest.raises(ValueError, match="exceed"):
        buffer.sample(batch_size=2)
