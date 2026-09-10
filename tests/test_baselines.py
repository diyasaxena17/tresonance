"""Tests for Phase 4 baseline forecasting and evaluation."""

import numpy as np
import pandas as pd

from market_resonance.data import MATURITIES
from market_resonance.evaluation import (
    evaluate_predictions,
    fit_linear_regression_baseline,
    flatten_sequence_features,
    regression_predictions,
    save_baseline_results,
    zero_change_predictions,
)
from market_resonance.evaluation.baselines import BaselineRun


def test_zero_change_predictions_match_target_shape() -> None:
    """The persistence baseline predicts a zero change for every target."""
    y_true = np.array([[0.01, -0.02], [0.03, 0.00]])

    prediction = zero_change_predictions(y_true)

    assert prediction.shape == y_true.shape
    np.testing.assert_allclose(prediction, 0.0)


def test_metrics_are_reported_in_basis_points() -> None:
    """A 0.01 percentage-point miss is one basis point."""
    y_true = np.full((2, len(MATURITIES)), 0.01)
    predictions = {"zero_change": np.zeros_like(y_true)}

    metrics = evaluate_predictions(y_true, predictions)
    overall = metrics[metrics["maturity"] == "overall"].iloc[0]

    assert overall["mae_bp"] == 1.0
    assert overall["rmse_bp"] == 1.0


def test_linear_regression_uses_flattened_sequence_features() -> None:
    """The tabular baseline maps flattened windows to multi-output targets."""
    X_train = np.arange(4 * 2 * 3, dtype=float).reshape(4, 2, 3)
    flat = flatten_sequence_features(X_train)
    y_train = np.column_stack([flat[:, 0] + 1.0, flat[:, -1] - 2.0])

    model = fit_linear_regression_baseline(X_train, y_train)
    prediction = regression_predictions(model, X_train)

    assert flat.shape == (4, 6)
    np.testing.assert_allclose(prediction, y_train)


def test_baseline_results_are_saved(tmp_path) -> None:
    """Metrics table and figure are written to requested paths."""
    metrics = pd.DataFrame(
        {
            "model": ["zero_change", "zero_change"],
            "maturity": ["overall", "3M"],
            "mae_bp": [1.0, 1.0],
            "rmse_bp": [2.0, 2.0],
        }
    )
    for maturity in MATURITIES[1:]:
        metrics.loc[len(metrics)] = ["zero_change", maturity, 1.0, 2.0]
    run = BaselineRun(
        metrics=metrics,
        predictions={"zero_change": np.zeros((1, len(MATURITIES)))},
        y_true=np.zeros((1, len(MATURITIES))),
        target_columns=[f"{maturity}_target" for maturity in MATURITIES],
    )

    table_path, figure_path = save_baseline_results(
        run,
        results_path=tmp_path / "baseline_results.csv",
        figure_path=tmp_path / "baseline_rmse.png",
    )

    assert table_path.exists()
    assert figure_path.exists()
