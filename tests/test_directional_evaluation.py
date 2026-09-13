"""Tests for shared directional forecast evaluation."""

import numpy as np
import pandas as pd
import pytest

from market_resonance.evaluation import (
    directional_metrics,
    directional_rewards,
    evaluate_directional_records,
    numerical_changes_to_directions,
)
from market_resonance.reinforcement import numerical_model_directional_records


def test_numerical_prediction_to_directional_class_conversion() -> None:
    changes = np.asarray([-0.02, -0.011, -0.010, 0.0, 0.010, 0.011, 0.02])

    directions = numerical_changes_to_directions(changes, threshold_bp=1.0)

    assert directions.tolist() == [0, 0, 1, 1, 1, 2, 2]


def test_threshold_boundary_behavior() -> None:
    directions = numerical_changes_to_directions(
        np.asarray([-0.01, 0.01]),
        threshold_bp=1.0,
    )

    assert directions.tolist() == [1, 1]


def test_directional_accuracy_and_reward_calculation() -> None:
    actual = np.asarray([0, 1, 2, 2])
    predicted = np.asarray([0, 2, 2, 1])

    metrics = directional_metrics(actual, predicted)
    rewards = directional_rewards(actual, predicted)

    assert metrics["direction_accuracy"] == pytest.approx(0.5)
    assert metrics["fall_recall"] == pytest.approx(1.0)
    assert metrics["flat_recall"] == pytest.approx(0.0)
    assert metrics["rise_recall"] == pytest.approx(0.5)
    assert metrics["average_directional_reward"] == pytest.approx(0.0)
    assert rewards.tolist() == [1, -1, 1, -1]


def test_common_date_alignment_limits_all_models_to_overlap() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    model_a = pd.DataFrame(
        {
            "model": "a",
            "date": dates,
            "actual_direction": [0, 1, 2],
            "predicted_direction": [0, 1, 1],
            "mae_bp": [1.0, 2.0, 3.0],
            "error_bp": [1.0, -2.0, 3.0],
        }
    )
    model_b = pd.DataFrame(
        {
            "model": "b",
            "date": dates[1:],
            "actual_direction": [1, 2],
            "predicted_direction": [1, 2],
            "mae_bp": [4.0, 5.0],
            "error_bp": [4.0, -5.0],
        }
    )

    evaluation = evaluate_directional_records(pd.concat([model_a, model_b]))

    assert evaluation.aligned_records["date"].nunique() == 2
    assert set(evaluation.aligned_records["date"]) == set(dates[1:])
    assert evaluation.summary.set_index("model").loc["a", "direction_accuracy"] == 0.5
    assert evaluation.summary.set_index("model").loc["b", "direction_accuracy"] == 1.0


def test_numerical_model_records_keep_regression_errors() -> None:
    records = numerical_model_directional_records(
        model_name="linear",
        dates=pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"])),
        actual_changes=np.asarray([0.02, -0.01]),
        predicted_changes=np.asarray([0.01, -0.03]),
        threshold_bp=1.0,
    )

    assert records["predicted_direction"].tolist() == [1, 0]
    assert records["actual_direction"].tolist() == [2, 1]
    assert records["mae_bp"].tolist() == pytest.approx([1.0, 2.0])
