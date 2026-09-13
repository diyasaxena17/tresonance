"""Smoke tests for Treasury directional DQN training."""

import numpy as np
import pandas as pd
import torch

from market_resonance.reinforcement import DQNTrainingConfig, train_dqn_agent


def test_dqn_training_executes_and_saves_checkpoint(tmp_path) -> None:
    frame = _training_frame()
    checkpoint_path = tmp_path / "treasury_direction_dqn.pt"

    run = train_dqn_agent(
        frame,
        config=DQNTrainingConfig(
            episodes=2,
            replay_capacity=64,
            batch_size=8,
            gamma=0.90,
            learning_rate=1e-3,
            epsilon_start=0.50,
            epsilon_end=0.10,
            target_update_steps=4,
            train_fraction=0.80,
            seed=123,
        ),
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint_path.exists()
    assert len(run.episode_rewards) == 2
    assert len(run.epsilon_history) == run.metadata["n_train_transitions"] * 2
    assert len(run.losses) > 0
    assert run.metadata["input_dim"] == 5
    assert run.metadata["n_train_rows"] == int(len(frame) * 0.80)
    assert "network.0.weight" in run.model_state_dict
    assert checkpoint["metadata"]["n_train_transitions"] == run.metadata[
        "n_train_transitions"
    ]


def test_dqn_training_is_seed_reproducible_without_checkpoint() -> None:
    frame = _training_frame()
    config = DQNTrainingConfig(
        episodes=1,
        replay_capacity=64,
        batch_size=8,
        train_fraction=0.80,
        seed=77,
    )

    first = train_dqn_agent(frame, config=config, checkpoint_path=None)
    second = train_dqn_agent(frame, config=config, checkpoint_path=None)

    assert first.episode_rewards == second.episode_rewards
    assert first.epsilon_history == second.epsilon_history
    np.testing.assert_allclose(first.losses, second.losses)


def _training_frame(rows: int = 64) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=rows)
    trend = np.linspace(0.0, 0.25, rows)
    wave = np.sin(np.arange(rows) / 3.0) * 0.02
    ten_year = 4.0 + trend + wave
    return pd.DataFrame(
        {
            "date": dates,
            "2Y": 3.7 + trend * 0.8,
            "10Y": ten_year,
        }
    )
