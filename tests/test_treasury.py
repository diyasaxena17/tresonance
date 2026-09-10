"""Tests for the Treasury data contract."""

import pandas as pd
import pytest

from market_resonance.data import (
    MATURITIES,
    DataValidationError,
    clean_treasury_yields,
)
from market_resonance.data.treasury import _fred_date_column


def _valid_frame() -> pd.DataFrame:
    """Return a small, deliberately unsorted valid Treasury table."""
    return pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02"],
            **{
                maturity: [4.0, 3.9]
                for maturity in MATURITIES
            },
        }
    )


def test_cleaning_sorts_dates_and_preserves_expected_columns() -> None:
    """Dates are ascending and columns use the fixed project contract."""
    cleaned = clean_treasury_yields(_valid_frame())

    assert cleaned.columns.tolist() == ["date", *MATURITIES]
    assert cleaned["date"].tolist() == list(
        pd.to_datetime(["2024-01-02", "2024-01-03"])
    )


def test_duplicate_dates_are_rejected() -> None:
    """Two observations for one date are ambiguous and must fail loudly."""
    frame = pd.concat([_valid_frame(), _valid_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(DataValidationError, match="duplicate dates"):
        clean_treasury_yields(frame)


def test_missing_yields_are_removed_without_forward_fill() -> None:
    """Incomplete rows are dropped instead of inventing an old observation."""
    frame = _valid_frame()
    frame.loc[0, "10Y"] = None

    cleaned = clean_treasury_yields(frame)

    assert len(cleaned) == 1
    assert cleaned["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]


def test_yield_columns_are_numeric() -> None:
    """Numeric strings from CSV input are converted to numeric values."""
    frame = _valid_frame().astype({maturity: str for maturity in MATURITIES})

    cleaned = clean_treasury_yields(frame)

    assert all(
        pd.api.types.is_numeric_dtype(cleaned[maturity]) for maturity in MATURITIES
    )


def test_missing_required_column_is_rejected() -> None:
    """A malformed source response cannot silently become a partial dataset."""
    frame = _valid_frame().drop(columns=["30Y"])

    with pytest.raises(DataValidationError, match="30Y"):
        clean_treasury_yields(frame)


def test_fred_observation_date_header_is_supported() -> None:
    """FRED's key-free graph CSV endpoint uses observation_date."""
    columns = pd.Index(["observation_date", "DGS10"])

    assert _fred_date_column(columns) == "observation_date"
