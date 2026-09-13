"""Directional metrics shared by regression forecasts and RL agents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASIS_POINTS_PER_PERCENTAGE_POINT = 100.0
ACTION_FALL = 0
ACTION_FLAT = 1
ACTION_RISE = 2
VALID_ACTIONS = (ACTION_FALL, ACTION_FLAT, ACTION_RISE)
DEFAULT_DIRECTION_THRESHOLD_BP = 1.0
ACTION_LABELS = {
    ACTION_FALL: "fall",
    ACTION_FLAT: "flat",
    ACTION_RISE: "rise",
}


@dataclass(frozen=True)
class DirectionalEvaluation:
    """Directional summary table plus confusion-matrix counts."""

    summary: pd.DataFrame
    confusion: pd.DataFrame
    aligned_records: pd.DataFrame


def numerical_changes_to_directions(
    changes: np.ndarray,
    threshold_bp: float = DEFAULT_DIRECTION_THRESHOLD_BP,
) -> np.ndarray:
    """Map numerical yield-change forecasts to FALL/FLAT/RISE classes."""
    if threshold_bp < 0:
        raise ValueError("threshold_bp must be non-negative.")

    values = np.asarray(changes, dtype=float)
    threshold = threshold_bp / BASIS_POINTS_PER_PERCENTAGE_POINT
    directions = np.full(values.shape, ACTION_FLAT, dtype=int)
    directions[values < -threshold] = ACTION_FALL
    directions[values > threshold] = ACTION_RISE
    return directions


def directional_rewards(
    actual_directions: np.ndarray,
    predicted_directions: np.ndarray,
) -> np.ndarray:
    """Return +1 for correct direction decisions and -1 otherwise."""
    actual, predicted = _validated_direction_arrays(
        actual_directions,
        predicted_directions,
    )
    return np.where(actual == predicted, 1, -1)


def directional_metrics(
    actual_directions: np.ndarray,
    predicted_directions: np.ndarray,
) -> dict[str, float]:
    """Compute overall accuracy, per-class recall, and average reward."""
    actual, predicted = _validated_direction_arrays(
        actual_directions,
        predicted_directions,
    )
    rewards = directional_rewards(actual, predicted)
    metrics = {
        "direction_accuracy": float(np.mean(actual == predicted)),
        "average_directional_reward": float(np.mean(rewards)),
    }
    for action, label in ACTION_LABELS.items():
        mask = actual == action
        metrics[f"{label}_recall"] = (
            float(np.mean(predicted[mask] == action)) if mask.any() else np.nan
        )
    return metrics


def confusion_matrix_counts(
    actual_directions: np.ndarray,
    predicted_directions: np.ndarray,
) -> pd.DataFrame:
    """Return count rows for each actual/predicted direction pair."""
    actual, predicted = _validated_direction_arrays(
        actual_directions,
        predicted_directions,
    )
    rows = []
    for actual_action, actual_label in ACTION_LABELS.items():
        for predicted_action, predicted_label in ACTION_LABELS.items():
            rows.append(
                {
                    "actual_direction": actual_label,
                    "predicted_direction": predicted_label,
                    "count": int(
                        np.sum(
                            (actual == actual_action)
                            & (predicted == predicted_action)
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def evaluate_directional_records(records: pd.DataFrame) -> DirectionalEvaluation:
    """Evaluate model/date directional records on their common date intersection.

    Expected columns are ``model``, ``date``, ``actual_direction``, and
    ``predicted_direction``. Optional ``mae_bp`` and ``rmse_bp`` fields are
    averaged per model when present. Dates not shared by every model are removed
    before any metrics are computed.
    """
    required = {"model", "date", "actual_direction", "predicted_direction"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    prepared = records.copy()
    prepared["date"] = pd.to_datetime(prepared["date"])
    model_count = prepared["model"].nunique()
    common_dates = (
        prepared.drop_duplicates(["model", "date"])
        .groupby("date")["model"]
        .nunique()
    )
    common_dates = common_dates[common_dates == model_count].index
    aligned = prepared[prepared["date"].isin(common_dates)].copy()
    if aligned.empty:
        raise ValueError("No common evaluation dates across models.")

    summary_rows = []
    confusion_frames = []
    for model_name, group in aligned.groupby("model", sort=False):
        metrics = directional_metrics(
            group["actual_direction"].to_numpy(),
            group["predicted_direction"].to_numpy(),
        )
        row = {
            "model": model_name,
            "mae_bp": _optional_mean(group, "mae_bp"),
            "rmse_bp": _optional_rmse_from_error(group),
            **metrics,
            "n_dates": int(group["date"].nunique()),
            "start_date": group["date"].min(),
            "end_date": group["date"].max(),
        }
        summary_rows.append(row)
        confusion = confusion_matrix_counts(
            group["actual_direction"].to_numpy(),
            group["predicted_direction"].to_numpy(),
        )
        confusion.insert(0, "model", model_name)
        confusion_frames.append(confusion)

    return DirectionalEvaluation(
        summary=pd.DataFrame(summary_rows),
        confusion=pd.concat(confusion_frames, ignore_index=True),
        aligned_records=aligned.reset_index(drop=True),
    )


def _optional_mean(group: pd.DataFrame, column: str) -> float:
    if column not in group:
        return np.nan
    values = group[column].dropna()
    return float(values.mean()) if not values.empty else np.nan


def _optional_rmse_from_error(group: pd.DataFrame) -> float:
    if "error_bp" not in group:
        return _optional_mean(group, "rmse_bp")
    values = group["error_bp"].dropna()
    return float(np.sqrt(np.mean(values.to_numpy(dtype=float) ** 2)))


def _validated_direction_arrays(
    actual_directions: np.ndarray,
    predicted_directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual_directions, dtype=int)
    predicted = np.asarray(predicted_directions, dtype=int)
    if actual.shape != predicted.shape:
        raise ValueError("actual and predicted directions must have the same shape.")
    if actual.size == 0:
        raise ValueError("direction arrays must not be empty.")
    valid = set(VALID_ACTIONS)
    if not set(np.unique(actual)).issubset(valid):
        raise ValueError("actual directions contain invalid action labels.")
    if not set(np.unique(predicted)).issubset(valid):
        raise ValueError("predicted directions contain invalid action labels.")
    return actual, predicted
