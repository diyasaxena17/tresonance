"""Exploratory analysis figures for Treasury yield data."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/market_resonance_mpl")

import matplotlib.pyplot as plt
import pandas as pd

from market_resonance.data import DEFAULT_OUTPUT_PATH, MATURITIES

DEFAULT_FIGURE_DIR = Path("reports/figures")
EXAMPLE_CURVE_DATES = (
    "1990-01-02",
    "2000-01-03",
    "2007-06-29",
    "2020-03-09",
    "2024-01-02",
)


def load_treasury_dataset(path: Path = DEFAULT_OUTPUT_PATH) -> pd.DataFrame:
    """Load the cleaned Treasury yield dataset."""
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.sort_values("date").reset_index(drop=True)


def add_curve_slope(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a 10Y-2Y slope column measured in percentage points."""
    with_slope = frame.copy()
    with_slope["10Y_minus_2Y"] = with_slope["10Y"] - with_slope["2Y"]
    return with_slope


def daily_yield_changes(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one-day yield changes in basis points."""
    changes = frame[list(MATURITIES)].diff().dropna() * 100
    changes.insert(0, "date", frame.loc[changes.index, "date"].to_numpy())
    return changes.reset_index(drop=True)


def _set_common_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
        }
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_historical_yields(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot each maturity's yield through time."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for maturity in MATURITIES:
        ax.plot(frame["date"], frame[maturity], linewidth=1.0, label=maturity)
    ax.set_title("Daily U.S. Treasury Constant-Maturity Yields")
    ax.set_xlabel("Date")
    ax.set_ylabel("Yield (%)")
    ax.legend(ncol=4, frameon=True)
    return _save(fig, output_dir / "treasury_historical_yields.png")


def plot_example_yield_curves(
    frame: pd.DataFrame,
    output_dir: Path,
    example_dates: Sequence[str] = EXAMPLE_CURVE_DATES,
) -> Path:
    """Plot cross-sectional yield curves for selected dates."""
    indexed = frame.set_index("date")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(MATURITIES))

    for date_text in example_dates:
        target = pd.Timestamp(date_text)
        nearest_index = indexed.index.get_indexer([target], method="nearest")[0]
        selected_date = indexed.index[nearest_index]
        ax.plot(
            x,
            indexed.iloc[nearest_index][list(MATURITIES)],
            marker="o",
            linewidth=1.8,
            label=selected_date.strftime("%Y-%m-%d"),
        )

    ax.set_title("Example Treasury Yield Curves")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Yield (%)")
    ax.set_xticks(list(x), MATURITIES)
    ax.legend(frameon=True)
    return _save(fig, output_dir / "treasury_example_yield_curves.png")


def plot_correlation_matrix(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot correlations among maturity yield levels."""
    corr = frame[list(MATURITIES)].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Correlation Across Treasury Maturities")
    ax.set_xticks(range(len(MATURITIES)), MATURITIES)
    ax.set_yticks(range(len(MATURITIES)), MATURITIES)
    for row in range(len(MATURITIES)):
        for col in range(len(MATURITIES)):
            ax.text(
                col,
                row,
                f"{corr.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    return _save(fig, output_dir / "treasury_maturity_correlation.png")


def plot_daily_change_distributions(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot distributions of one-day yield changes in basis points."""
    changes = daily_yield_changes(frame)
    fig, axes = plt.subplots(4, 2, figsize=(10, 9), sharex=True)
    axes_flat = axes.ravel()

    for axis, maturity in zip(axes_flat, MATURITIES, strict=False):
        axis.hist(changes[maturity], bins=80, color="#3b6ea8", alpha=0.85)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(maturity)
        axis.set_ylabel("Days")

    axes_flat[-1].axis("off")
    for axis in axes[-1, :]:
        axis.set_xlabel("Daily change (basis points)")

    fig.suptitle("Distributions of Daily Treasury Yield Changes", y=1.01)
    return _save(fig, output_dir / "treasury_daily_change_distributions.png")


def plot_curve_slope(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot the 10Y-2Y Treasury yield-curve slope."""
    with_slope = add_curve_slope(frame)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(
        with_slope["date"],
        with_slope["10Y_minus_2Y"],
        linewidth=1.1,
        color="#226f54",
    )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.fill_between(
        with_slope["date"],
        with_slope["10Y_minus_2Y"],
        0,
        where=with_slope["10Y_minus_2Y"] < 0,
        color="#c44536",
        alpha=0.25,
        interpolate=True,
        label="Inversion",
    )
    ax.set_title("Treasury Curve Slope: 10Y Minus 2Y")
    ax.set_xlabel("Date")
    ax.set_ylabel("Percentage points")
    ax.legend(frameon=True)
    return _save(fig, output_dir / "treasury_10y_minus_2y_slope.png")


def generate_treasury_eda_figures(
    data_path: Path = DEFAULT_OUTPUT_PATH,
    output_dir: Path = DEFAULT_FIGURE_DIR,
) -> list[Path]:
    """Generate all exploratory Treasury figures."""
    _set_common_style()
    frame = load_treasury_dataset(data_path)
    return [
        plot_historical_yields(frame, output_dir),
        plot_example_yield_curves(frame, output_dir),
        plot_correlation_matrix(frame, output_dir),
        plot_daily_change_distributions(frame, output_dir),
        plot_curve_slope(frame, output_dir),
    ]
