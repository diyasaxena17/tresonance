"""Tests for the lightweight Treasury directional RL environment."""

import numpy as np
import pandas as pd
import pytest

from market_resonance.reinforcement import (
    ACTION_FALL,
    ACTION_FLAT,
    ACTION_RISE,
    TreasuryDirectionEnv,
    realized_direction,
)


def _rl_frame(rows: int = 26) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=rows)
    ten_year = np.full(rows, 4.00)
    ten_year[21:] = [4.00, 3.98, 3.985, 4.00, 4.02][: rows - 21]
    return pd.DataFrame(
        {
            "date": dates,
            "2Y": np.linspace(3.50, 3.75, rows),
            "10Y": ten_year,
        }
    )


def test_realized_direction_labels_use_basis_point_threshold() -> None:
    assert realized_direction(-0.011, threshold_bp=1.0) == ACTION_FALL
    assert realized_direction(-0.010, threshold_bp=1.0) == ACTION_FLAT
    assert realized_direction(0.000, threshold_bp=1.0) == ACTION_FLAT
    assert realized_direction(0.010, threshold_bp=1.0) == ACTION_FLAT
    assert realized_direction(0.011, threshold_bp=1.0) == ACTION_RISE


def test_reward_calculation_and_info_for_matching_action() -> None:
    env = TreasuryDirectionEnv(_rl_frame())
    state = env.reset()

    next_state, reward, done, info = env.step(ACTION_FALL)

    assert state.tolist() == pytest.approx([4.0, 0.0, 0.0, 0.0, 0.29])
    assert reward == 1
    assert done is False
    assert next_state is not None
    assert info["realized_direction"] == ACTION_FALL
    assert info["next_day_10y_change"] == pytest.approx(-0.02)


def test_reward_is_negative_for_incorrect_direction() -> None:
    env = TreasuryDirectionEnv(_rl_frame())
    env.reset()

    _, reward, _, info = env.step(ACTION_RISE)

    assert reward == -1
    assert info["realized_direction"] == ACTION_FALL


def test_chronological_stepping_and_reset_behavior() -> None:
    env = TreasuryDirectionEnv(_rl_frame())
    first_state = env.reset()

    next_state, _, _, first_info = env.step(ACTION_FALL)
    assert env.current_date == first_info["next_date"]
    _, _, _, second_info = env.step(ACTION_FLAT)
    reset_state = env.reset()

    assert first_info["date"] < first_info["next_date"]
    assert first_info["next_date"] == second_info["date"]
    assert env.current_date == first_info["date"]
    assert next_state.tolist() == pytest.approx([3.98, -0.02, -0.02, -0.02, 0.26])
    np.testing.assert_allclose(reset_state, first_state)


def test_terminal_behavior() -> None:
    env = TreasuryDirectionEnv(_rl_frame())
    env.reset()

    done = False
    state = None
    for _ in range(env.n_transitions):
        state, _, done, _ = env.step(ACTION_FLAT)

    assert done is True
    assert state is None
    with pytest.raises(RuntimeError, match="episode is done"):
        env.step(ACTION_FLAT)


def test_current_state_does_not_use_future_values() -> None:
    frame = _rl_frame()
    baseline_state = TreasuryDirectionEnv(frame).reset()
    mutated = frame.copy()
    mutated.loc[22, "10Y"] = 99.0
    mutated_state = TreasuryDirectionEnv(mutated).reset()

    np.testing.assert_allclose(mutated_state, baseline_state)


def test_malformed_input_handling() -> None:
    with pytest.raises(ValueError, match="Missing required columns"):
        TreasuryDirectionEnv(pd.DataFrame({"date": ["2024-01-01"], "10Y": [4.0]}))

    malformed = _rl_frame().astype({"10Y": object})
    malformed.loc[0, "10Y"] = "not-a-number"
    with pytest.raises(ValueError, match="complete and numeric"):
        TreasuryDirectionEnv(malformed)

    with pytest.raises(ValueError, match="threshold_bp"):
        TreasuryDirectionEnv(_rl_frame(), threshold_bp=-1.0)


def test_invalid_action_is_rejected() -> None:
    env = TreasuryDirectionEnv(_rl_frame())
    env.reset()

    with pytest.raises(ValueError, match="action must be"):
        env.step(99)
