"""Minimal DQN training loop for Treasury direction decisions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from market_resonance.reinforcement.dqn import TreasuryDQN
from market_resonance.reinforcement.environment import (
    DEFAULT_DIRECTION_THRESHOLD_BP,
    TreasuryDirectionEnv,
)
from market_resonance.reinforcement.replay_buffer import ReplayBuffer, TransitionBatch

DEFAULT_DQN_CHECKPOINT_PATH = Path("reports/models/treasury_direction_dqn.pt")


@dataclass(frozen=True)
class DQNTrainingConfig:
    """Beginner-friendly training settings for the Treasury DQN."""

    episodes: int = 5
    replay_capacity: int = 10_000
    batch_size: int = 32
    gamma: float = 0.95
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    target_update_steps: int = 100
    train_fraction: float = 0.70
    threshold_bp: float = DEFAULT_DIRECTION_THRESHOLD_BP
    seed: int = 42


@dataclass(frozen=True)
class DQNTrainingRun:
    """Trained DQN state and compact training history."""

    model_state_dict: dict[str, torch.Tensor]
    episode_rewards: list[float]
    losses: list[float]
    epsilon_history: list[float]
    metadata: dict


def train_dqn_agent(
    frame: pd.DataFrame,
    config: DQNTrainingConfig | None = None,
    checkpoint_path: Path | None = DEFAULT_DQN_CHECKPOINT_PATH,
) -> DQNTrainingRun:
    """Train a small DQN only on the chronological training period.

    The raw Treasury frame is sorted by date and split chronologically before
    any environment is created. Only the training slice is used to generate
    transitions, so replay sampling can randomize experience order without ever
    mixing validation/test-period observations into the buffer.
    """
    if config is None:
        config = DQNTrainingConfig()
    _validate_config(config)
    _set_deterministic_seed(config.seed)

    train_frame = _chronological_train_frame(frame, config.train_fraction)
    env = TreasuryDirectionEnv(train_frame, threshold_bp=config.threshold_bp)
    initial_state = env.reset()
    input_dim = int(initial_state.shape[0])

    online_model = TreasuryDQN(input_dim=input_dim, seed=config.seed)
    target_model = TreasuryDQN(input_dim=input_dim, seed=config.seed)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(online_model.parameters(), lr=config.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    replay_buffer = ReplayBuffer(capacity=config.replay_capacity, seed=config.seed)

    episode_rewards: list[float] = []
    losses: list[float] = []
    epsilon_history: list[float] = []
    optimization_steps = 0
    total_decision_steps = max(1, config.episodes * env.n_transitions)

    for _episode in range(config.episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            epsilon = _linear_epsilon(
                step=len(epsilon_history),
                total_steps=total_decision_steps,
                start=config.epsilon_start,
                end=config.epsilon_end,
            )
            action = online_model.epsilon_greedy_action(
                torch.as_tensor(state, dtype=torch.float32),
                epsilon=epsilon,
            )
            next_state, reward, done, _ = env.step(action)
            replay_buffer.append(state, action, reward, next_state, done)

            episode_reward += reward
            epsilon_history.append(epsilon)
            state = next_state if next_state is not None else state

            if len(replay_buffer) >= config.batch_size:
                batch = replay_buffer.sample(config.batch_size)
                loss = _optimization_step(
                    batch=batch,
                    online_model=online_model,
                    target_model=target_model,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    gamma=config.gamma,
                )
                losses.append(loss)
                optimization_steps += 1

                if optimization_steps % config.target_update_steps == 0:
                    target_model.load_state_dict(online_model.state_dict())

        episode_rewards.append(episode_reward)

    metadata = {
        "config": config.__dict__,
        "input_dim": input_dim,
        "n_train_rows": int(len(train_frame)),
        "n_train_transitions": int(env.n_transitions),
        "optimization_steps": optimization_steps,
        "feature_columns": list(env.feature_columns),
    }
    run = DQNTrainingRun(
        model_state_dict=online_model.state_dict(),
        episode_rewards=episode_rewards,
        losses=losses,
        epsilon_history=epsilon_history,
        metadata=metadata,
    )
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": run.model_state_dict,
                "episode_rewards": run.episode_rewards,
                "losses": run.losses,
                "epsilon_history": run.epsilon_history,
                "metadata": run.metadata,
            },
            checkpoint_path,
        )
    return run


def _optimization_step(
    batch: TransitionBatch,
    online_model: TreasuryDQN,
    target_model: TreasuryDQN,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    gamma: float,
) -> float:
    q_values = online_model(batch.states)
    chosen_q_values = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q_values = target_model(batch.next_states).max(dim=1).values
        next_q_values = torch.where(
            batch.dones,
            torch.zeros_like(next_q_values),
            next_q_values,
        )
        targets = batch.rewards + gamma * next_q_values

    loss = loss_fn(chosen_q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def _chronological_train_frame(
    frame: pd.DataFrame,
    train_fraction: float,
) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ValueError("frame must contain a date column.")
    sorted_frame = frame.copy()
    sorted_frame["date"] = pd.to_datetime(sorted_frame["date"], errors="coerce")
    if sorted_frame["date"].isna().any():
        raise ValueError("The date column contains invalid dates.")
    sorted_frame = sorted_frame.sort_values("date").reset_index(drop=True)
    train_rows = int(len(sorted_frame) * train_fraction)
    if train_rows < 1:
        raise ValueError("train_fraction produced an empty training period.")
    return sorted_frame.iloc[:train_rows].reset_index(drop=True)


def _linear_epsilon(step: int, total_steps: int, start: float, end: float) -> float:
    if total_steps <= 1:
        return end
    progress = min(step / (total_steps - 1), 1.0)
    return start + progress * (end - start)


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _validate_config(config: DQNTrainingConfig) -> None:
    if config.episodes < 1:
        raise ValueError("episodes must be at least 1.")
    if config.replay_capacity < 1:
        raise ValueError("replay_capacity must be at least 1.")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if not 0.0 <= config.gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if not 0.0 <= config.epsilon_end <= config.epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1.")
    if config.target_update_steps < 1:
        raise ValueError("target_update_steps must be at least 1.")
    if not 0 < config.train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
