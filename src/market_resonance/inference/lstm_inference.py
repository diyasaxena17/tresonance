"""Inference path for the saved LSTM Treasury forecaster."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from market_resonance.data import DEFAULT_OUTPUT_PATH, MATURITIES
from market_resonance.features import CHANGE_LAGS, create_treasury_features
from market_resonance.models import YieldCurveLSTM
from market_resonance.training import DEFAULT_CHECKPOINT_PATH

DEFAULT_INFERENCE_METRICS_PATH = Path("results/inference_metrics.json")
BASIS_POINTS_PER_PERCENTAGE_POINT = 100.0


@dataclass(frozen=True)
class InferenceResult:
    """Forecast and runtime diagnostics from LSTM inference."""

    forecast_date: str
    latest_input_date: str
    forecast: list[dict[str, float | str]]
    parameter_count: int
    single_sample_latency_ms: float
    batch_size: int
    batch_latency_ms: float


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def load_model_from_checkpoint(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> tuple[YieldCurveLSTM, dict]:
    """Load an LSTM model and inference metadata from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    metadata = checkpoint["metadata"]
    required_metadata = {
        "normalization_mean",
        "normalization_scale",
        "feature_columns",
        "target_columns",
        "lookback",
        "horizon",
    }
    missing_metadata = required_metadata.difference(metadata)
    if missing_metadata:
        raise ValueError(
            "Checkpoint is missing inference metadata: "
            f"{', '.join(sorted(missing_metadata))}"
        )

    model = YieldCurveLSTM(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, metadata


def run_lstm_inference(
    data_path: Path = DEFAULT_OUTPUT_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    metrics_path: Path = DEFAULT_INFERENCE_METRICS_PATH,
    batch_size: int = 64,
    repeats: int = 25,
) -> InferenceResult:
    """Run inference on the latest available 60-day sequence and save metrics."""
    model, metadata = load_model_from_checkpoint(checkpoint_path)
    yields = pd.read_csv(data_path, parse_dates=["date"])
    latest_sequence, latest_yields, latest_date = _latest_normalized_sequence(
        yields,
        metadata,
    )
    X_single = torch.as_tensor(latest_sequence[None, :, :], dtype=torch.float32)
    X_batch = X_single.repeat(batch_size, 1, 1)

    with torch.no_grad():
        single_prediction = model(X_single)
        single_latency_ms = _measure_latency_ms(model, X_single, repeats=repeats)
        batch_latency_ms = _measure_latency_ms(model, X_batch, repeats=repeats)

    predicted_changes = single_prediction.squeeze(0).numpy()
    forecast = _format_forecast(
        latest_yields=latest_yields,
        predicted_changes=predicted_changes,
    )
    result = InferenceResult(
        forecast_date=f"{latest_date.date()} + {metadata['horizon']} trading day(s)",
        latest_input_date=latest_date.strftime("%Y-%m-%d"),
        forecast=forecast,
        parameter_count=count_parameters(model),
        single_sample_latency_ms=round(single_latency_ms, 4),
        batch_size=batch_size,
        batch_latency_ms=round(batch_latency_ms, 4),
    )
    _save_metrics(result, metrics_path)
    return result


def _latest_normalized_sequence(
    yields: pd.DataFrame,
    metadata: dict,
) -> tuple[np.ndarray, pd.Series, pd.Timestamp]:
    lookback = int(metadata["lookback"])
    feature_columns = metadata["feature_columns"]
    features = create_treasury_features(yields, change_lags=CHANGE_LAGS)
    clean_features = features.dropna(subset=feature_columns).reset_index(drop=True)
    if len(clean_features) < lookback:
        raise ValueError(
            f"Need at least {lookback} complete feature rows for inference."
        )

    latest_window = clean_features.loc[
        len(clean_features) - lookback :,
        feature_columns,
    ].to_numpy(dtype=float)
    mean = np.asarray(metadata["normalization_mean"], dtype=float)
    scale = np.asarray(metadata["normalization_scale"], dtype=float)
    normalized = (latest_window - mean.squeeze(0)) / scale.squeeze(0)

    latest_date = clean_features["date"].iloc[-1]
    yield_row = yields[yields["date"] == latest_date].iloc[-1]
    latest_yields = yield_row[list(MATURITIES)].astype(float)
    return normalized, latest_yields, latest_date


def _measure_latency_ms(
    model: YieldCurveLSTM,
    X: torch.Tensor,
    repeats: int,
) -> float:
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            model(X)
    elapsed = time.perf_counter() - start
    return elapsed * 1000 / repeats


def _format_forecast(
    latest_yields: pd.Series,
    predicted_changes: np.ndarray,
) -> list[dict[str, float | str]]:
    rows = []
    for maturity_index, maturity in enumerate(MATURITIES):
        change_percentage_points = float(predicted_changes[maturity_index])
        latest_yield = float(latest_yields[maturity])
        rows.append(
            {
                "maturity": maturity,
                "latest_yield_percent": round(latest_yield, 4),
                "predicted_change_bp": round(
                    change_percentage_points * BASIS_POINTS_PER_PERCENTAGE_POINT,
                    4,
                ),
                "forecast_yield_percent": round(
                    latest_yield + change_percentage_points,
                    4,
                ),
            }
        )
    return rows


def _save_metrics(result: InferenceResult, metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "forecast_date": result.forecast_date,
                "latest_input_date": result.latest_input_date,
                "forecast": result.forecast,
                "parameter_count": result.parameter_count,
                "single_sample_latency_ms": result.single_sample_latency_ms,
                "batch_size": result.batch_size,
                "batch_latency_ms": result.batch_latency_ms,
            },
            indent=2,
        )
        + "\n"
    )
