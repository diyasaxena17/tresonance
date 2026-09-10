"""Download, clean, and validate daily Treasury constant-maturity yields."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

MATURITIES = ("3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y")
FRED_SERIES = {
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "5Y": "DGS5",
    "10Y": "DGS10",
    "30Y": "DGS30",
}
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
DEFAULT_OUTPUT_PATH = Path("data/processed/treasury_yields_daily.csv")
FRED_DATE_COLUMNS = ("observation_date", "DATE")


class DataValidationError(ValueError):
    """Raised when downloaded Treasury data violates the data contract."""


def _fred_date_column(columns: pd.Index) -> str:
    """Return the date column used by a FRED CSV response."""
    for column in FRED_DATE_COLUMNS:
        if column in columns:
            return column
    raise DataValidationError(
        "FRED response did not contain an observation_date or DATE column."
    )


def _fred_url(
    series_id: str,
    start_date: str | None,
    end_date: str | None,
) -> str:
    """Build a key-free FRED CSV URL for one series."""
    parameters = [f"id={series_id}"]
    if start_date is not None:
        parameters.append(f"cosd={start_date}")
    if end_date is not None:
        parameters.append(f"coed={end_date}")
    return f"{FRED_CSV_URL}?{'&'.join(parameters)}"


def clean_treasury_yields(frame: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate a Treasury yield table.

    The returned table has one row per date and these columns, in order:
    ``date`` followed by the seven maturities. Rows with any missing yield are
    removed rather than forward-filled.

    Args:
        frame: A table with a ``date`` column and one column per maturity.

    Returns:
        A chronologically sorted table with numeric, complete yield columns.

    Raises:
        DataValidationError: If required columns, dates, or uniqueness rules
            are violated.
    """
    expected_columns = ["date", *MATURITIES]
    missing_columns = [
        column for column in expected_columns if column not in frame.columns
    ]
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    cleaned = frame.loc[:, expected_columns].copy()
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    if cleaned["date"].isna().any():
        raise DataValidationError("The date column contains invalid dates.")
    cleaned["date"] = cleaned["date"].dt.normalize()

    if cleaned["date"].duplicated().any():
        raise DataValidationError("The dataset contains duplicate dates.")

    for maturity in MATURITIES:
        cleaned[maturity] = pd.to_numeric(cleaned[maturity], errors="coerce")

    cleaned = cleaned.dropna(subset=list(MATURITIES))
    cleaned = cleaned.sort_values("date").reset_index(drop=True)

    if cleaned.empty:
        raise DataValidationError(
            "No complete observations remain after removing missing yields."
        )

    return cleaned


def download_treasury_yields(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download and save daily Treasury constant-maturity yields from FRED.

    FRED's public CSV endpoint requires no API key. The source series are
    DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10, and DGS30. The saved CSV contains
    only complete observations shared by all seven series.

    Args:
        output_path: Destination for the cleaned CSV dataset.
        start_date: Optional inclusive date in ``YYYY-MM-DD`` format.
        end_date: Optional inclusive date in ``YYYY-MM-DD`` format.

    Returns:
        The same cleaned DataFrame written to ``output_path``.
    """
    series_frames = []
    for maturity in MATURITIES:
        series_id = FRED_SERIES[maturity]
        url = _fred_url(series_id, start_date, end_date)
        series = pd.read_csv(url, na_values=["."])
        date_column = _fred_date_column(series.columns)
        if series_id not in series.columns:
            raise DataValidationError(
                f"FRED response for {series_id} did not contain {series_id}."
            )
        series_frames.append(
            series.loc[:, [date_column, series_id]].rename(
                columns={date_column: "date", series_id: maturity}
            )
        )

    combined = series_frames[0]
    for series in series_frames[1:]:
        combined = combined.merge(series, on="date", how="outer")

    cleaned = clean_treasury_yields(combined)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    return cleaned
