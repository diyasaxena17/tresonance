"""Tests for Treasury feature engineering and supervised windows."""

import numpy as np
import pandas as pd
import pytest

from market_resonance.data import MATURITIES
from market_resonance.features import (
    build_supervised_windows,
    chronological_train_validation_test_split,
    create_treasury_features,
    fit_standardizer,
    standardize_splits,
)


def _synthetic_yields(rows: int = 120) -> pd.DataFrame:
    """Create deterministic yield data with maturity-specific slopes."""
    dates = pd.bdate_range("2024-01-01", periods=rows)
    frame = pd.DataFrame({"date": dates})
    for maturity_index, maturity in enumerate(MATURITIES, start=1):
        frame[maturity] = maturity_index * 10.0 + np.arange(rows) * maturity_index
    return frame


def test_feature_columns_and_trailing_changes_are_created() -> None:
    """Each maturity has level, 1-day, 5-day, and 21-day change features."""
    frame = _synthetic_yields()

    features = create_treasury_features(frame)

    assert features.shape[1] == 1 + 7 * 4
    assert "10Y_level" in features.columns
    assert "10Y_chg_1d" in features.columns
    assert "10Y_chg_5d" in features.columns
    assert "10Y_chg_21d" in features.columns
    assert features.loc[21, "10Y_chg_21d"] == pytest.approx(126.0)


def test_supervised_window_tensor_shapes_and_alignment() -> None:
    """The first sample uses previous 60 rows and targets the next row."""
    frame = _synthetic_yields()

    windows = build_supervised_windows(frame, lookback=60, horizon=1)

    assert windows.X.shape == (39, 60, 28)
    assert windows.y.shape == (39, 7)
    assert windows.X[0, -1, windows.feature_columns.index("2Y_level")] == 360.0
    assert windows.y[0].tolist() == pytest.approx([1, 2, 3, 4, 5, 6, 7])
    assert windows.sample_start_dates.iloc[0] == frame.loc[21, "date"]
    assert windows.sample_end_dates.iloc[0] == frame.loc[80, "date"]
    assert windows.target_dates.iloc[0] == frame.loc[81, "date"]


def test_horizon_is_configurable() -> None:
    """A 5-day horizon changes targets from t to t+5."""
    frame = _synthetic_yields()

    windows = build_supervised_windows(frame, lookback=60, horizon=5)

    assert windows.y[0].tolist() == pytest.approx([5, 10, 15, 20, 25, 30, 35])
    assert windows.target_columns[0] == "3M_target_chg_5d"


def test_chronological_split_preserves_sample_order() -> None:
    """Train, validation, and test examples remain ordered in time."""
    windows = build_supervised_windows(_synthetic_yields(), lookback=60, horizon=1)

    splits = chronological_train_validation_test_split(
        windows,
        train_fraction=0.60,
        validation_fraction=0.20,
    )

    assert splits.train.sample_end_dates.is_monotonic_increasing
    assert splits.validation.sample_end_dates.is_monotonic_increasing
    assert splits.test.sample_end_dates.is_monotonic_increasing
    assert (
        splits.train.sample_end_dates.iloc[-1]
        < splits.validation.sample_end_dates.iloc[0]
    )
    assert (
        splits.validation.sample_end_dates.iloc[-1]
        < splits.test.sample_end_dates.iloc[0]
    )


def test_future_row_between_origin_and_target_does_not_enter_current_sample() -> None:
    """For horizon 2, row t+1 cannot affect sample ending at t."""
    frame = _synthetic_yields()
    baseline = build_supervised_windows(frame, lookback=60, horizon=2)
    mutated = frame.copy()
    mutated.loc[81, "10Y"] = 999_999.0

    changed = build_supervised_windows(mutated, lookback=60, horizon=2)

    np.testing.assert_allclose(changed.X[0], baseline.X[0])
    np.testing.assert_allclose(changed.y[0], baseline.y[0])


def test_target_row_change_affects_y_but_not_X() -> None:
    """The future target row is label data, never input data."""
    frame = _synthetic_yields()
    baseline = build_supervised_windows(frame, lookback=60, horizon=1)
    mutated = frame.copy()
    mutated.loc[81, "10Y"] = 999_999.0

    changed = build_supervised_windows(mutated, lookback=60, horizon=1)

    np.testing.assert_allclose(changed.X[0], baseline.X[0])
    assert changed.y[0, list(MATURITIES).index("10Y")] != baseline.y[0, 5]


def test_standardizer_is_fit_on_training_data_only() -> None:
    """Validation and test values cannot change fitted normalization parameters."""
    windows = build_supervised_windows(_synthetic_yields(180), lookback=60, horizon=1)
    splits = chronological_train_validation_test_split(windows)
    baseline = fit_standardizer(splits.train.X)

    polluted_validation = splits.validation.X.copy()
    polluted_validation[:] = 1_000_000.0
    polluted_test = splits.test.X.copy()
    polluted_test[:] = -1_000_000.0

    polluted_splits = type(splits)(
        train=splits.train,
        validation=type(splits.validation)(
            X=polluted_validation,
            y=splits.validation.y,
            feature_columns=splits.validation.feature_columns,
            target_columns=splits.validation.target_columns,
            sample_start_dates=splits.validation.sample_start_dates,
            sample_end_dates=splits.validation.sample_end_dates,
            target_dates=splits.validation.target_dates,
        ),
        test=type(splits.test)(
            X=polluted_test,
            y=splits.test.y,
            feature_columns=splits.test.feature_columns,
            target_columns=splits.test.target_columns,
            sample_start_dates=splits.test.sample_start_dates,
            sample_end_dates=splits.test.sample_end_dates,
            target_dates=splits.test.target_dates,
        ),
    )

    _, polluted_standardizer = standardize_splits(polluted_splits)

    np.testing.assert_allclose(polluted_standardizer.mean_, baseline.mean_)
    np.testing.assert_allclose(polluted_standardizer.scale_, baseline.scale_)
