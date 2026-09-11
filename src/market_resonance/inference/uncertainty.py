"""Simple residual-covariance uncertainty for LSTM yield forecasts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/market_resonance_mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from market_resonance.data import DEFAULT_OUTPUT_PATH, MATURITIES
from market_resonance.features import (
    build_supervised_windows,
    chronological_train_validation_test_split,
    standardize_splits,
)
from market_resonance.inference.lstm_inference import (
    BASIS_POINTS_PER_PERCENTAGE_POINT,
    _latest_normalized_sequence,
    load_model_from_checkpoint,
)

DEFAULT_UNCERTAINTY_SUMMARY_PATH = Path("results/uncertainty_summary.json")
DEFAULT_SCENARIO_PATH = Path("results/monte_carlo_yield_curves.csv")
DEFAULT_RANGE_PATH = Path("reports/tables/uncertainty_ranges.csv")
DEFAULT_FAN_FIGURE_PATH = Path("reports/figures/yield_curve_fan.png")
DEFAULT_RANGE_FIGURE_PATH = Path("reports/figures/uncertainty_ranges_by_maturity.png")


@dataclass(frozen=True)
class UncertaintyRun:
    """Outputs from Monte Carlo uncertainty inference."""

    summary_path: Path
    scenario_path: Path
    range_path: Path
    fan_figure_path: Path
    range_figure_path: Path


def run_uncertainty_simulation(
    data_path: Path = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path = Path("reports/models/multi_frequency_lstm.pt"),
    summary_path: Path = DEFAULT_UNCERTAINTY_SUMMARY_PATH,
    scenario_path: Path = DEFAULT_SCENARIO_PATH,
    range_path: Path = DEFAULT_RANGE_PATH,
    fan_figure_path: Path = DEFAULT_FAN_FIGURE_PATH,
    range_figure_path: Path = DEFAULT_RANGE_FIGURE_PATH,
    scenario_count: int = 1_000,
    seed: int = 42,
) -> UncertaintyRun:
    """Generate Monte Carlo yield-curve scenarios around the neural forecast."""
    rng = np.random.default_rng(seed)
    model, metadata = load_model_from_checkpoint(checkpoint_path)
    yields = pd.read_csv(data_path, parse_dates=["date"])

    residuals = _validation_residuals(yields, model, metadata)
    covariance = np.cov(residuals, rowvar=False)
    covariance = _stabilize_covariance(covariance)

    latest_sequence, latest_yields, latest_date = _latest_normalized_sequence(
        yields,
        metadata,
    )
    point_change = _predict(model, latest_sequence[None, :, :]).squeeze(0)
    point_yields = latest_yields.to_numpy(dtype=float) + point_change
    sampled_errors = rng.multivariate_normal(
        mean=np.zeros(len(MATURITIES)),
        cov=covariance,
        size=scenario_count,
    )
    simulated_yields = point_yields[None, :] + sampled_errors

    scenario_frame = pd.DataFrame(simulated_yields, columns=MATURITIES)
    scenario_frame.insert(0, "scenario", np.arange(1, scenario_count + 1))
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_frame.to_csv(scenario_path, index=False)

    ranges = _confidence_ranges(point_yields, simulated_yields)
    range_path.parent.mkdir(parents=True, exist_ok=True)
    ranges.to_csv(range_path, index=False)

    _save_fan_figure(point_yields, simulated_yields, fan_figure_path)
    _save_range_figure(ranges, range_figure_path)

    summary = {
        "latest_input_date": latest_date.strftime("%Y-%m-%d"),
        "scenario_count": scenario_count,
        "residual_source": "validation split only",
        "assumptions": [
            "Validation residuals are representative of next forecast errors.",
            "Forecast errors are centered at zero around the neural point forecast.",
            "Forecast errors follow a multivariate normal distribution.",
            "The covariance estimate is fixed for this one-step forecast.",
        ],
        "residual_covariance_basis_points": (
            covariance * BASIS_POINTS_PER_PERCENTAGE_POINT**2
        ).round(6).tolist(),
        "point_forecast_yields_percent": {
            maturity: round(float(point_yields[index]), 6)
            for index, maturity in enumerate(MATURITIES)
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    return UncertaintyRun(
        summary_path=summary_path,
        scenario_path=scenario_path,
        range_path=range_path,
        fan_figure_path=fan_figure_path,
        range_figure_path=range_figure_path,
    )


def _validation_residuals(
    yields: pd.DataFrame,
    model: torch.nn.Module,
    metadata: dict,
) -> np.ndarray:
    windows = build_supervised_windows(
        yields,
        lookback=int(metadata["lookback"]),
        horizon=int(metadata["horizon"]),
        change_lags=tuple(metadata["change_lags"]),
    )
    splits = chronological_train_validation_test_split(windows)
    standardized_splits, _ = standardize_splits(splits)
    predictions = _predict(model, standardized_splits.validation.X)
    return standardized_splits.validation.y - predictions


def _predict(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(X, dtype=torch.float32)
        return model(tensor).numpy()


def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
    return covariance + np.eye(covariance.shape[0]) * 1e-10


def _confidence_ranges(
    point_yields: np.ndarray,
    simulated_yields: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for index, maturity in enumerate(MATURITIES):
        values = simulated_yields[:, index]
        rows.append(
            {
                "maturity": maturity,
                "point_forecast_yield_percent": round(float(point_yields[index]), 6),
                "p05_yield_percent": round(float(np.percentile(values, 5)), 6),
                "p50_yield_percent": round(float(np.percentile(values, 50)), 6),
                "p95_yield_percent": round(float(np.percentile(values, 95)), 6),
                "width_90pct_bp": round(
                    float(
                        (np.percentile(values, 95) - np.percentile(values, 5))
                        * BASIS_POINTS_PER_PERCENTAGE_POINT
                    ),
                    6,
                ),
            }
        )
    return pd.DataFrame(rows)


def _save_fan_figure(
    point_yields: np.ndarray,
    simulated_yields: np.ndarray,
    figure_path: Path,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(MATURITIES))
    p05 = np.percentile(simulated_yields, 5, axis=0)
    p25 = np.percentile(simulated_yields, 25, axis=0)
    p75 = np.percentile(simulated_yields, 75, axis=0)
    p95 = np.percentile(simulated_yields, 95, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(x, p05, p95, alpha=0.20, label="5th-95th percentile")
    ax.fill_between(x, p25, p75, alpha=0.35, label="25th-75th percentile")
    ax.plot(x, point_yields, marker="o", linewidth=2, label="Point forecast")
    ax.set_title("Monte Carlo Yield-Curve Fan")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Yield (%)")
    ax.set_xticks(x, MATURITIES)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_range_figure(ranges: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(ranges["maturity"], ranges["width_90pct_bp"], color="#3b6ea8")
    ax.set_title("90% Monte Carlo Range Width by Maturity")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Width (basis points)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
