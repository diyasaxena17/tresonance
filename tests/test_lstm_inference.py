"""Tests for saved-model LSTM inference."""

import json

import numpy as np
import pandas as pd
import torch

from market_resonance.data import MATURITIES
from market_resonance.inference import count_parameters, run_lstm_inference
from market_resonance.models import YieldCurveLSTM


def test_lstm_inference_saves_metrics_json(tmp_path) -> None:
    """Inference loads checkpoint metadata and writes forecast diagnostics."""
    data_path = tmp_path / "yields.csv"
    checkpoint_path = tmp_path / "checkpoint.pt"
    metrics_path = tmp_path / "inference_metrics.json"
    _synthetic_yields(100).to_csv(data_path, index=False)
    model = YieldCurveLSTM(input_size=28, hidden_size=8, output_size=7)
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "validation_loss": 0.1,
            "config": {},
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
                "lookback": 60,
                "horizon": 1,
            },
        },
        checkpoint_path,
    )

    result = run_lstm_inference(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        batch_size=4,
        repeats=2,
    )
    saved = json.loads(metrics_path.read_text())

    assert len(result.forecast) == 7
    assert saved["parameter_count"] == count_parameters(model)
    assert saved["batch_size"] == 4
    assert saved["single_sample_latency_ms"] >= 0
    assert saved["batch_latency_ms"] >= 0


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
