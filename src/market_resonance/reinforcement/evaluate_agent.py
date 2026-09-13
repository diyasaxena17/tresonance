"""Directional evaluation helpers for the Treasury DQN agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from market_resonance.data import DEFAULT_OUTPUT_PATH, MATURITIES
from market_resonance.evaluation.directional import (
    BASIS_POINTS_PER_PERCENTAGE_POINT,
    DirectionalEvaluation,
    evaluate_directional_records,
    numerical_changes_to_directions,
)
from market_resonance.features import (
    build_supervised_windows,
    chronological_train_validation_test_split,
    standardize_splits,
)
from market_resonance.models import YieldCurveLSTM
from market_resonance.reinforcement.dqn import TreasuryDQN
from market_resonance.reinforcement.environment import (
    DEFAULT_DIRECTION_THRESHOLD_BP,
    TreasuryDirectionEnv,
)

TEN_YEAR_INDEX = list(MATURITIES).index("10Y")
DEFAULT_DAILY_LSTM_CHECKPOINT = Path("reports/models/daily_lstm.pt")
DEFAULT_MULTI_FREQUENCY_LSTM_CHECKPOINT = Path("reports/models/multi_frequency_lstm.pt")


@dataclass(frozen=True)
class DirectionalComparisonRun:
    """Common-date directional comparison outputs."""

    evaluation: DirectionalEvaluation
    compatibility_notes: list[str]


def dqn_directional_records(
    model: TreasuryDQN,
    frame: pd.DataFrame,
    model_name: str = "dqn",
    threshold_bp: float = DEFAULT_DIRECTION_THRESHOLD_BP,
) -> pd.DataFrame:
    """Evaluate a DQN over one chronological frame and return dated actions."""
    env = TreasuryDirectionEnv(frame, threshold_bp=threshold_bp)
    state = env.reset()
    rows = []
    done = False
    while not done:
        action = model.greedy_action(torch.as_tensor(state, dtype=torch.float32))
        next_state, _reward, done, info = env.step(action)
        actual = numerical_changes_to_directions(
            np.asarray([info["next_day_10y_change"]]),
            threshold_bp=threshold_bp,
        )[0]
        rows.append(
            {
                "model": model_name,
                "date": info["next_date"],
                "actual_change": info["next_day_10y_change"],
                "actual_direction": int(actual),
                "predicted_direction": int(action),
                "predicted_change": np.nan,
                "mae_bp": np.nan,
                "error_bp": np.nan,
            }
        )
        if next_state is not None:
            state = next_state
    return pd.DataFrame(rows)


def numerical_model_directional_records(
    model_name: str,
    dates: pd.Series,
    actual_changes: np.ndarray,
    predicted_changes: np.ndarray,
    threshold_bp: float = DEFAULT_DIRECTION_THRESHOLD_BP,
) -> pd.DataFrame:
    """Convert dated 10Y numerical predictions into directional records."""
    actual = np.asarray(actual_changes, dtype=float)
    predicted = np.asarray(predicted_changes, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("actual_changes and predicted_changes must have same shape.")
    if len(dates) != actual.shape[0]:
        raise ValueError("dates length must match prediction length.")

    actual_direction = numerical_changes_to_directions(actual, threshold_bp)
    predicted_direction = numerical_changes_to_directions(predicted, threshold_bp)
    error_bp = (predicted - actual) * BASIS_POINTS_PER_PERCENTAGE_POINT
    return pd.DataFrame(
        {
            "model": model_name,
            "date": pd.to_datetime(dates),
            "actual_change": actual,
            "predicted_change": predicted,
            "actual_direction": actual_direction,
            "predicted_direction": predicted_direction,
            "mae_bp": np.abs(error_bp),
            "error_bp": error_bp,
        }
    )


def compare_directional_models(
    records: list[pd.DataFrame],
) -> DirectionalComparisonRun:
    """Evaluate all supplied models on the common overlapping date set."""
    combined = pd.concat(records, ignore_index=True)
    evaluation = evaluate_directional_records(combined)
    notes = []
    if len(evaluation.aligned_records) < len(combined):
        notes.append(
            "Model records did not cover identical dates; metrics use the "
            "common overlapping date intersection only."
        )
    return DirectionalComparisonRun(evaluation=evaluation, compatibility_notes=notes)


def lstm_checkpoint_10y_records(
    checkpoint_path: Path,
    data_path: Path = DEFAULT_OUTPUT_PATH,
    model_name: str | None = None,
    threshold_bp: float = DEFAULT_DIRECTION_THRESHOLD_BP,
) -> pd.DataFrame:
    """Regenerate dated 10Y test predictions from a saved LSTM checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint["metadata"]
    change_lags = tuple(metadata.get("change_lags", (1, 5, 21)))
    lookback = int(metadata["lookback"])
    horizon = int(metadata["horizon"])
    if horizon != 1:
        raise ValueError("Directional comparison currently expects horizon=1.")

    yields = pd.read_csv(data_path, parse_dates=["date"])
    windows = build_supervised_windows(
        yields,
        lookback=lookback,
        horizon=horizon,
        change_lags=change_lags,
    )
    splits = chronological_train_validation_test_split(windows)
    standardized_splits, _ = standardize_splits(splits)

    model = YieldCurveLSTM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        predictions = model(
            torch.as_tensor(standardized_splits.test.X, dtype=torch.float32)
        ).numpy()

    name = model_name or metadata.get("feature_set") or checkpoint_path.stem
    return numerical_model_directional_records(
        model_name=name,
        dates=standardized_splits.test.target_dates,
        actual_changes=standardized_splits.test.y[:, TEN_YEAR_INDEX],
        predicted_changes=predictions[:, TEN_YEAR_INDEX],
        threshold_bp=threshold_bp,
    )


def baseline_10y_records(
    data_path: Path = DEFAULT_OUTPUT_PATH,
    threshold_bp: float = DEFAULT_DIRECTION_THRESHOLD_BP,
) -> list[pd.DataFrame]:
    """Regenerate dated 10Y records for persistence and linear regression."""
    from market_resonance.evaluation.baselines import (
        fit_linear_regression_baseline,
        regression_predictions,
        zero_change_predictions,
    )

    yields = pd.read_csv(data_path, parse_dates=["date"])
    windows = build_supervised_windows(yields)
    splits = chronological_train_validation_test_split(windows)
    standardized_splits, _ = standardize_splits(splits)
    linear_model = fit_linear_regression_baseline(
        standardized_splits.train.X,
        standardized_splits.train.y,
    )
    predictions = {
        "persistence_zero_change": zero_change_predictions(standardized_splits.test.y),
        "linear_regression": regression_predictions(
            linear_model,
            standardized_splits.test.X,
        ),
    }
    return [
        numerical_model_directional_records(
            model_name=model_name,
            dates=standardized_splits.test.target_dates,
            actual_changes=standardized_splits.test.y[:, TEN_YEAR_INDEX],
            predicted_changes=prediction[:, TEN_YEAR_INDEX],
            threshold_bp=threshold_bp,
        )
        for model_name, prediction in predictions.items()
    ]
