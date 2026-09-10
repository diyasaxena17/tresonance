"""Feature engineering and supervised sequence windows for Treasury yields.

This module converts the cleaned daily yield table into model-ready X/y arrays.
It also owns chronological splits and train-only normalization to prevent leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from market_resonance.data import MATURITIES

CHANGE_LAGS = (1, 5, 21)
DEFAULT_LOOKBACK_DAYS = 60
DEFAULT_HORIZON_DAYS = 1


@dataclass(frozen=True)
class SupervisedWindows:
    """Batch-first supervised windows for sequence forecasting."""

    X: np.ndarray
    y: np.ndarray
    feature_columns: list[str]
    target_columns: list[str]
    sample_start_dates: pd.Series
    sample_end_dates: pd.Series
    target_dates: pd.Series


@dataclass(frozen=True)
class WindowSplits:
    """Chronological train, validation, and test windows."""

    train: SupervisedWindows
    validation: SupervisedWindows
    test: SupervisedWindows


@dataclass(frozen=True)
class Standardizer:
    """Feature-wise standardization parameters fitted on training inputs only."""

    mean_: np.ndarray
    scale_: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the frozen standardization parameters."""
        return (X - self.mean_) / self.scale_


def treasury_feature_columns(
    maturities: tuple[str, ...] = MATURITIES,
    change_lags: tuple[int, ...] = CHANGE_LAGS,
) -> list[str]:
    """Return feature column names in fixed maturity-major order."""
    columns = []
    for maturity in maturities:
        columns.append(f"{maturity}_level")
        columns.extend(f"{maturity}_chg_{lag}d" for lag in change_lags)
    return columns


def target_columns(
    horizon: int = DEFAULT_HORIZON_DAYS,
    maturities: tuple[str, ...] = MATURITIES,
) -> list[str]:
    """Return target column names for a forecast horizon."""
    return [f"{maturity}_target_chg_{horizon}d" for maturity in maturities]


def create_treasury_features(
    frame: pd.DataFrame,
    maturities: tuple[str, ...] = MATURITIES,
    change_lags: tuple[int, ...] = CHANGE_LAGS,
) -> pd.DataFrame:
    """Create yield-level and trailing-change features for every maturity.

    Change features are current yield minus the yield from ``lag`` trading rows
    earlier. They never use future rows.
    """
    expected_columns = ["date", *maturities]
    missing_columns = [column for column in expected_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    sorted_frame = frame.loc[:, expected_columns].copy()
    sorted_frame["date"] = pd.to_datetime(sorted_frame["date"])
    sorted_frame = sorted_frame.sort_values("date").reset_index(drop=True)

    features = pd.DataFrame({"date": sorted_frame["date"]})
    for maturity in maturities:
        features[f"{maturity}_level"] = sorted_frame[maturity].astype(float)
        for lag in change_lags:
            features[f"{maturity}_chg_{lag}d"] = (
                sorted_frame[maturity].astype(float)
                - sorted_frame[maturity].astype(float).shift(lag)
            )
    return features


def build_supervised_windows(
    frame: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK_DAYS,
    horizon: int = DEFAULT_HORIZON_DAYS,
    maturities: tuple[str, ...] = MATURITIES,
    change_lags: tuple[int, ...] = CHANGE_LAGS,
) -> SupervisedWindows:
    """Build supervised sequence windows and future yield-change targets.

    Each sample uses ``lookback`` consecutive feature rows ending at date ``t``.
    Its target is the vector of maturity yield changes from ``t`` to
    ``t + horizon``.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")

    sorted_frame = frame.loc[:, ["date", *maturities]].copy()
    sorted_frame["date"] = pd.to_datetime(sorted_frame["date"])
    sorted_frame = sorted_frame.sort_values("date").reset_index(drop=True)

    features = create_treasury_features(sorted_frame, maturities, change_lags)
    feature_cols = treasury_feature_columns(maturities, change_lags)
    feature_matrix = features[feature_cols].to_numpy(dtype=float)
    level_matrix = sorted_frame[list(maturities)].to_numpy(dtype=float)

    max_lag = max(change_lags, default=0)
    first_end_index = max_lag + lookback - 1
    last_end_index = len(sorted_frame) - horizon - 1
    if first_end_index > last_end_index:
        raise ValueError(
            "Not enough observations to create one supervised window with the "
            f"requested lookback={lookback} and horizon={horizon}."
        )

    X_values = []
    y_values = []
    sample_start_dates = []
    sample_end_dates = []
    future_target_dates = []

    for end_index in range(first_end_index, last_end_index + 1):
        start_index = end_index - lookback + 1
        window = feature_matrix[start_index : end_index + 1]
        if np.isnan(window).any():
            raise ValueError("Feature window contains missing values.")

        X_values.append(window)
        y_values.append(level_matrix[end_index + horizon] - level_matrix[end_index])
        sample_start_dates.append(sorted_frame.loc[start_index, "date"])
        sample_end_dates.append(sorted_frame.loc[end_index, "date"])
        future_target_dates.append(sorted_frame.loc[end_index + horizon, "date"])

    return SupervisedWindows(
        X=np.stack(X_values),
        y=np.stack(y_values),
        feature_columns=feature_cols,
        target_columns=target_columns(horizon, maturities),
        sample_start_dates=pd.Series(sample_start_dates, name="sample_start_date"),
        sample_end_dates=pd.Series(sample_end_dates, name="sample_end_date"),
        target_dates=pd.Series(future_target_dates, name="target_date"),
    )


def _slice_windows(
    windows: SupervisedWindows,
    start: int,
    stop: int,
) -> SupervisedWindows:
    return SupervisedWindows(
        X=windows.X[start:stop],
        y=windows.y[start:stop],
        feature_columns=windows.feature_columns,
        target_columns=windows.target_columns,
        sample_start_dates=windows.sample_start_dates.iloc[start:stop].reset_index(
            drop=True
        ),
        sample_end_dates=windows.sample_end_dates.iloc[start:stop].reset_index(
            drop=True
        ),
        target_dates=windows.target_dates.iloc[start:stop].reset_index(drop=True),
    )


def chronological_train_validation_test_split(
    windows: SupervisedWindows,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> WindowSplits:
    """Split windows chronologically without shuffling."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split.")

    sample_count = windows.X.shape[0]
    train_stop = int(sample_count * train_fraction)
    validation_stop = train_stop + int(sample_count * validation_fraction)
    has_empty_split = (
        train_stop == 0
        or validation_stop == train_stop
        or validation_stop == sample_count
    )
    if has_empty_split:
        raise ValueError("Split fractions produced an empty split.")

    return WindowSplits(
        train=_slice_windows(windows, 0, train_stop),
        validation=_slice_windows(windows, train_stop, validation_stop),
        test=_slice_windows(windows, validation_stop, sample_count),
    )


def fit_standardizer(X_train: np.ndarray) -> Standardizer:
    """Fit feature-wise normalization on training windows only."""
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    scale = X_train.std(axis=(0, 1), keepdims=True)
    scale = np.where(scale == 0, 1.0, scale)
    return Standardizer(mean_=mean, scale_=scale)


def standardize_splits(splits: WindowSplits) -> tuple[WindowSplits, Standardizer]:
    """Standardize X arrays using parameters fitted only on train.X."""
    standardizer = fit_standardizer(splits.train.X)
    standardized = WindowSplits(
        train=_replace_X(splits.train, standardizer.transform(splits.train.X)),
        validation=_replace_X(
            splits.validation,
            standardizer.transform(splits.validation.X),
        ),
        test=_replace_X(splits.test, standardizer.transform(splits.test.X)),
    )
    return standardized, standardizer


def _replace_X(windows: SupervisedWindows, X: np.ndarray) -> SupervisedWindows:
    return SupervisedWindows(
        X=X,
        y=windows.y,
        feature_columns=windows.feature_columns,
        target_columns=windows.target_columns,
        sample_start_dates=windows.sample_start_dates,
        sample_end_dates=windows.sample_end_dates,
        target_dates=windows.target_dates,
    )
