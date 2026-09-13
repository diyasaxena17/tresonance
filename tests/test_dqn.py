"""Tests for the Treasury directional DQN."""

import pytest
import torch

from market_resonance.reinforcement import (
    ACTION_FALL,
    ACTION_FLAT,
    ACTION_RISE,
    TreasuryDQN,
)


def test_dqn_forward_output_shape_and_action_values() -> None:
    model = TreasuryDQN(input_dim=5, seed=123)
    X = torch.zeros((4, 5), dtype=torch.float32)

    q_values = model(X)
    shapes = model.shape_trace(X)

    assert q_values.shape == (4, 3)
    assert q_values.shape[1] == 3
    assert shapes == {
        "input": (4, 5),
        "hidden_1": (4, 32),
        "activated_1": (4, 32),
        "hidden_2": (4, 32),
        "activated_2": (4, 32),
        "q_values": (4, 3),
    }


def test_greedy_action_selects_highest_q_value() -> None:
    model = TreasuryDQN(input_dim=5, seed=123)
    _force_constant_q_values(model, [-1.0, 0.5, 2.0])

    action = model.greedy_action(torch.zeros(5))

    assert action == ACTION_RISE


def test_epsilon_zero_uses_greedy_action() -> None:
    model = TreasuryDQN(input_dim=5, seed=123)
    _force_constant_q_values(model, [3.0, 1.0, -1.0])

    actions = [
        model.epsilon_greedy_action(torch.zeros(5), epsilon=0.0) for _ in range(5)
    ]

    assert actions == [ACTION_FALL] * 5


def test_epsilon_one_uses_exploration_path() -> None:
    model = TreasuryDQN(input_dim=5, seed=123)
    _force_constant_q_values(model, [3.0, 1.0, -1.0])

    actions = [
        model.epsilon_greedy_action(torch.zeros(5), epsilon=1.0) for _ in range(10)
    ]

    assert set(actions).issubset({ACTION_FALL, ACTION_FLAT, ACTION_RISE})
    assert any(action != ACTION_FALL for action in actions)


def test_seeded_reproducibility_for_weights_and_exploration() -> None:
    first = TreasuryDQN(input_dim=5, seed=77)
    second = TreasuryDQN(input_dim=5, seed=77)
    X = torch.ones((2, 5), dtype=torch.float32)

    torch.testing.assert_close(first(X), second(X))
    first_actions = [
        first.epsilon_greedy_action(torch.ones(5), epsilon=1.0) for _ in range(6)
    ]
    second_actions = [
        second.epsilon_greedy_action(torch.ones(5), epsilon=1.0) for _ in range(6)
    ]

    assert first_actions == second_actions


def test_tensor_shape_validation() -> None:
    model = TreasuryDQN(input_dim=5, seed=123)

    with pytest.raises(ValueError, match="2D tensor"):
        model(torch.zeros(5))
    with pytest.raises(ValueError, match="batch, 5"):
        model(torch.zeros((2, 4)))
    with pytest.raises(ValueError, match="1D tensor"):
        model.greedy_action(torch.zeros((1, 5)))
    with pytest.raises(ValueError, match="epsilon"):
        model.epsilon_greedy_action(torch.zeros(5), epsilon=1.1)


def _force_constant_q_values(model: TreasuryDQN, q_values: list[float]) -> None:
    for parameter in model.parameters():
        parameter.data.zero_()
    output_layer = model.network[-1]
    output_layer.bias.data = torch.tensor(q_values, dtype=torch.float32)
