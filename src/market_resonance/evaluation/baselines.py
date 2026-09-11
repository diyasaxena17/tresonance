"""Baseline forecasting models and metrics for Treasury yield changes."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/market_resonance_mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
except ModuleNotFoundError:
    LinearRegression = None

from market_resonance.data import DEFAULT_OUTPUT_PATH, MATURITIES
from market_resonance.features import (
    build_supervised_windows,
    chronological_train_validation_test_split,
    standardize_splits,
)

DEFAULT_RESULTS_PATH = Path("reports/tables/baseline_results.csv")
DEFAULT_FIGURE_PATH = Path("reports/figures/baseline_rmse_by_maturity.png")
BASIS_POINTS_PER_PERCENTAGE_POINT = 100.0


@dataclass(frozen=True)
class BaselineRun:
    """Predictions, metrics, and metadata from baseline evaluation."""

    metrics: pd.DataFrame
    predictions: dict[str, np.ndarray]
    y_true: np.ndarray
    target_columns: list[str]


@dataclass(frozen=True)
class LinearRegressionBaseline:
    """Small multi-output ordinary least-squares fallback regression baseline."""

    coefficients: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict targets from a 2D design matrix."""
        design = _add_intercept_column(X)
        return design @ self.coefficients


def zero_change_predictions(y_true: np.ndarray) -> np.ndarray:
    """Predict no future yield change for every maturity."""
    return np.zeros_like(y_true, dtype=float)


def flatten_sequence_features(X: np.ndarray) -> np.ndarray:
    """Flatten batch-first sequence tensors for tabular models."""
    return X.reshape(X.shape[0], X.shape[1] * X.shape[2])


def fit_linear_regression_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> LinearRegression | LinearRegressionBaseline:
    """Fit a simple multi-output linear regression baseline."""
    if LinearRegression is not None:
        model = LinearRegression()
        model.fit(flatten_sequence_features(X_train), y_train)
        return model

    design = _add_intercept_column(flatten_sequence_features(X_train))
    coefficients, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    return LinearRegressionBaseline(coefficients=coefficients)


def regression_predictions(
    model: LinearRegression | LinearRegressionBaseline,
    X: np.ndarray,
) -> np.ndarray:
    """Predict future yield changes from flattened sequence features."""
    return model.predict(flatten_sequence_features(X))


def _add_intercept_column(X: np.ndarray) -> np.ndarray:
    """Add a leading intercept column to a 2D design matrix."""
    intercept = np.ones((X.shape[0], 1), dtype=X.dtype)
    return np.column_stack([intercept, X])


def evaluate_predictions(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    maturities: tuple[str, ...] = MATURITIES,
) -> pd.DataFrame:
    """Compute MAE and RMSE in basis points overall and by maturity."""
    rows = []
    y_true_bp = y_true * BASIS_POINTS_PER_PERCENTAGE_POINT

    for model_name, prediction in predictions.items():
        prediction_bp = prediction * BASIS_POINTS_PER_PERCENTAGE_POINT
        errors = prediction_bp - y_true_bp
        rows.append(
            {
                "model": model_name,
                "maturity": "overall",
                "mae_bp": np.mean(np.abs(errors)),
                "rmse_bp": np.sqrt(np.mean(errors**2)),
            }
        )
        for maturity_index, maturity in enumerate(maturities):
            maturity_errors = errors[:, maturity_index]
            rows.append(
                {
                    "model": model_name,
                    "maturity": maturity,
                    "mae_bp": np.mean(np.abs(maturity_errors)),
                    "rmse_bp": np.sqrt(np.mean(maturity_errors**2)),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics["mae_bp"] = metrics["mae_bp"].round(3)
    metrics["rmse_bp"] = metrics["rmse_bp"].round(3)
    return metrics


def run_baseline_evaluation(
    data_path: Path = DEFAULT_OUTPUT_PATH,
    lookback: int = 60,
    horizon: int = 1,
    change_lags: tuple[int, ...] = (1, 5, 21),
) -> BaselineRun:
    """Train and evaluate zero-change and linear-regression baselines."""
    yields = pd.read_csv(data_path, parse_dates=["date"])
    windows = build_supervised_windows(
        yields,
        lookback=lookback,
        horizon=horizon,
        change_lags=change_lags,
    )
    splits = chronological_train_validation_test_split(windows)
    standardized_splits, _ = standardize_splits(splits)

    linear_model = fit_linear_regression_baseline(
        standardized_splits.train.X,
        standardized_splits.train.y,
    )
    predictions = {
        "zero_change": zero_change_predictions(standardized_splits.test.y),
        "linear_regression": regression_predictions(
            linear_model,
            standardized_splits.test.X,
        ),
    }
    metrics = evaluate_predictions(
        standardized_splits.test.y,
        predictions,
    )
    return BaselineRun(
        metrics=metrics,
        predictions=predictions,
        y_true=standardized_splits.test.y,
        target_columns=standardized_splits.test.target_columns,
    )


def save_baseline_results(
    run: BaselineRun,
    results_path: Path = DEFAULT_RESULTS_PATH,
    figure_path: Path = DEFAULT_FIGURE_PATH,
) -> tuple[Path, Path]:
    """Save the baseline metrics table and RMSE-by-maturity figure."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    run.metrics.to_csv(results_path, index=False)

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    maturity_metrics = run.metrics[run.metrics["maturity"] != "overall"]
    pivot = maturity_metrics.pivot(
        index="maturity",
        columns="model",
        values="rmse_bp",
    ).loc[list(MATURITIES)]

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_title("Baseline Forecast Error by Maturity")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("RMSE (basis points)")
    ax.legend(title="Model", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return results_path, figure_path
