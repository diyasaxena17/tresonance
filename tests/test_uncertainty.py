"""Tests for stochastic uncertainty simulation."""

import json

import numpy as np
import pandas as pd
import torch

from market_resonance.data import MATURITIES
from market_resonance.inference.uncertainty import run_uncertainty_simulation
from market_resonance.models import YieldCurveLSTM


def test_uncertainty_simulation_saves_scenarios_and_ranges(tmp_path) -> None:
    """Monte Carlo uncertainty writes scenarios, ranges, and covariance summary."""
    data_path = tmp_path / "yields.csv"
    checkpoint_path = tmp_path / "model.pt"
    summary_path = tmp_path / "summary.json"
    scenario_path = tmp_path / "scenarios.csv"
    range_path = tmp_path / "ranges.csv"
    fan_path = tmp_path / "fan.png"
    range_figure_path = tmp_path / "ranges.png"
    distribution_figure_path = tmp_path / "distributions.png"
    simulation_figure_path = tmp_path / "simulations.png"
    _synthetic_yields(130).to_csv(data_path, index=False)
    model = YieldCurveLSTM(input_size=28, hidden_size=8, output_size=7)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_size": 28,
                "hidden_size": 8,
                "output_size": 7,
            },
            "metadata": {
                "normalization_mean": np.zeros((1, 1, 28)).tolist(),
                "normalization_scale": np.ones((1, 1, 28)).tolist(),
                "feature_columns": _feature_columns(),
                "target_columns": [
                    f"{maturity}_target_chg_1d" for maturity in MATURITIES
                ],
                "change_lags": (1, 5, 21),
                "lookback": 60,
                "horizon": 1,
            },
        },
        checkpoint_path,
    )

    run = run_uncertainty_simulation(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
        scenario_path=scenario_path,
        range_path=range_path,
        fan_figure_path=fan_path,
        range_figure_path=range_figure_path,
        distribution_figure_path=distribution_figure_path,
        simulation_figure_path=simulation_figure_path,
        scenario_count=25,
        seed=1,
    )
    summary = json.loads(summary_path.read_text())
    scenarios = pd.read_csv(scenario_path)
    ranges = pd.read_csv(range_path)

    assert run.summary_path.exists()
    assert scenarios.shape == (25, 8)
    assert len(ranges) == 7
    assert len(summary["residual_covariance_basis_points"]) == 7
    assert summary["residual_source"] == "validation split only"
    assert run.distribution_figure_path.exists()
    assert run.simulation_figure_path.exists()


def _synthetic_yields(rows: int) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=rows)
    frame = pd.DataFrame({"date": dates})
    for maturity_index, maturity in enumerate(MATURITIES, start=1):
        frame[maturity] = maturity_index + np.arange(rows) * 0.01
    return frame


def _feature_columns() -> list[str]:
    columns = []
    for maturity in MATURITIES:
        columns.append(f"{maturity}_level")
        columns.extend(
            [
                f"{maturity}_chg_1d",
                f"{maturity}_chg_5d",
                f"{maturity}_chg_21d",
            ]
        )
    return columns
