"""Tests for Treasury exploratory-analysis calculations."""

import pandas as pd
import pytest

from market_resonance.visualization import add_curve_slope, daily_yield_changes


def test_curve_slope_is_10y_minus_2y() -> None:
    """The slope helper measures the standard 10Y-2Y spread."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "2Y": [4.25],
            "10Y": [4.00],
        }
    )

    with_slope = add_curve_slope(frame)

    assert with_slope["10Y_minus_2Y"].tolist() == [-0.25]


def test_daily_yield_changes_are_basis_points() -> None:
    """A 0.10 percentage-point yield move equals 10 basis points."""
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "3M": [5.00, 5.10],
            "6M": [5.00, 5.10],
            "1Y": [5.00, 5.10],
            "2Y": [4.00, 4.10],
            "5Y": [3.80, 3.90],
            "10Y": [3.70, 3.80],
            "30Y": [3.60, 3.70],
        }
    )

    changes = daily_yield_changes(frame)

    assert len(changes) == 1
    assert changes["10Y"].iloc[0] == pytest.approx(10.0)
