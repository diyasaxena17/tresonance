"""Multi-frequency ablation study for LSTM Treasury forecasts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from market_resonance.data import DEFAULT_OUTPUT_PATH, MATURITIES
from market_resonance.evaluation import (
    evaluate_predictions,
    fit_linear_regression_baseline,
    regression_predictions,
    zero_change_predictions,
)
from market_resonance.features import (
    build_supervised_windows,
    chronological_train_validation_test_split,
    standardize_splits,
)
from market_resonance.models import YieldCurveLSTM
from market_resonance.training import (
    TrainingConfig,
    make_dataloaders,
    set_deterministic_seed,
    train_lstm_model,
)

DEFAULT_ABLATION_TABLE_PATH = Path("reports/tables/ablation_results.csv")
DEFAULT_ABLATION_SUMMARY_PATH = Path("results/ablation_summary.json")
DEFAULT_ABLATION_FIGURE_PATH = Path("reports/figures/ablation_rmse_by_maturity.png")
DEFAULT_ABLATION_LOSS_FIGURE_PATH = Path("reports/figures/ablation_validation_loss.png")


@dataclass(frozen=True)
class AblationSpec:
    """Feature-set definition for one ablation arm."""

    model_name: str
    change_lags: tuple[int, ...]


@dataclass(frozen=True)
class AblationRun:
    """Saved outputs and metrics from an ablation study."""

    metrics: pd.DataFrame
    summary: dict
    table_path: Path
    figure_path: Path
    loss_figure_path: Path


ABLATION_SPECS = (
    AblationSpec(model_name="daily_lstm", change_lags=(1,)),
    AblationSpec(model_name="multi_frequency_lstm", change_lags=(1, 5, 21)),
)


def run_ablation_study(
    data_path: Path = DEFAULT_OUTPUT_PATH,
    table_path: Path = DEFAULT_ABLATION_TABLE_PATH,
    summary_path: Path = DEFAULT_ABLATION_SUMMARY_PATH,
    figure_path: Path = DEFAULT_ABLATION_FIGURE_PATH,
    loss_figure_path: Path = DEFAULT_ABLATION_LOSS_FIGURE_PATH,
    lookback: int = 60,
    horizon: int = 1,
    hidden_size: int = 64,
    config: TrainingConfig | None = None,
) -> AblationRun:
    """Train matched daily and multi-frequency LSTMs and compare results."""
    if config is None:
        config = TrainingConfig(max_epochs=10, patience=3, seed=42)

    yields = pd.read_csv(data_path, parse_dates=["date"])
    metric_frames = []
    histories = {}

    for spec in ABLATION_SPECS:
        set_deterministic_seed(config.seed)
        windows = build_supervised_windows(
            yields,
            lookback=lookback,
            horizon=horizon,
            change_lags=spec.change_lags,
        )
        splits = chronological_train_validation_test_split(windows)
        standardized_splits, standardizer = standardize_splits(splits)
        train_loader, validation_loader, _ = make_dataloaders(
            standardized_splits,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        model = YieldCurveLSTM(
            input_size=standardized_splits.train.X.shape[-1],
            hidden_size=hidden_size,
            output_size=standardized_splits.train.y.shape[-1],
        )
        history = train_lstm_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            config=config,
            checkpoint_path=Path(f"reports/models/{spec.model_name}.pt"),
            checkpoint_metadata={
                "feature_set": spec.model_name,
                "change_lags": spec.change_lags,
                "normalization_mean": standardizer.mean_.tolist(),
                "normalization_scale": standardizer.scale_.tolist(),
                "feature_columns": standardized_splits.train.feature_columns,
                "target_columns": standardized_splits.train.target_columns,
                "lookback": lookback,
                "horizon": horizon,
            },
        )
        _load_best_model(model, Path(f"reports/models/{spec.model_name}.pt"))
        histories[spec.model_name] = {
            "train_loss": history.train_loss,
            "validation_loss": history.validation_loss,
            "best_epoch": history.best_epoch,
            "best_validation_loss": history.best_validation_loss,
        }

        metric_frames.append(
            _evaluate_split_models(
                split_name="validation",
                model_name=spec.model_name,
                model=model,
                X=standardized_splits.validation.X,
                y=standardized_splits.validation.y,
            )
        )
        metric_frames.append(
            _evaluate_split_models(
                split_name="test",
                model_name=spec.model_name,
                model=model,
                X=standardized_splits.test.X,
                y=standardized_splits.test.y,
            )
        )
        metric_frames.append(
            _evaluate_baselines_for_split(
                split_name="validation",
                train_X=standardized_splits.train.X,
                train_y=standardized_splits.train.y,
                X=standardized_splits.validation.X,
                y=standardized_splits.validation.y,
                feature_set=spec.model_name,
            )
        )
        metric_frames.append(
            _evaluate_baselines_for_split(
                split_name="test",
                train_X=standardized_splits.train.X,
                train_y=standardized_splits.train.y,
                X=standardized_splits.test.X,
                y=standardized_splits.test.y,
                feature_set=spec.model_name,
            )
        )

    metrics = pd.concat(metric_frames, ignore_index=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(table_path, index=False)

    _save_rmse_figure(metrics, figure_path)
    _save_loss_figure(histories, loss_figure_path)
    summary = _build_summary(metrics, histories)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    return AblationRun(
        metrics=metrics,
        summary=summary,
        table_path=table_path,
        figure_path=figure_path,
        loss_figure_path=loss_figure_path,
    )


def _evaluate_split_models(
    split_name: str,
    model_name: str,
    model: YieldCurveLSTM,
    X: np.ndarray,
    y: np.ndarray,
) -> pd.DataFrame:
    predictions = _predict_lstm(model, X)
    metrics = evaluate_predictions(y, {model_name: predictions})
    metrics.insert(0, "split", split_name)
    metrics.insert(1, "feature_set", model_name)
    return metrics


def _evaluate_baselines_for_split(
    split_name: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    feature_set: str,
) -> pd.DataFrame:
    linear_model = fit_linear_regression_baseline(train_X, train_y)
    predictions = {
        "persistence_zero_change": zero_change_predictions(y),
        "linear_regression": regression_predictions(linear_model, X),
    }
    metrics = evaluate_predictions(y, predictions)
    metrics.insert(0, "split", split_name)
    metrics.insert(1, "feature_set", feature_set)
    return metrics


def _predict_lstm(model: YieldCurveLSTM, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(X, dtype=torch.float32)
        return model(tensor).numpy()


def _load_best_model(model: YieldCurveLSTM, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()


def _save_rmse_figure(metrics: pd.DataFrame, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    test_metrics = metrics[
        (metrics["split"] == "test") & (metrics["maturity"] != "overall")
    ]
    selected = test_metrics[
        test_metrics["model"].isin(
            [
                "daily_lstm",
                "multi_frequency_lstm",
                "persistence_zero_change",
                "linear_regression",
            ]
        )
    ]
    pivot = selected.pivot_table(
        index="maturity",
        columns="model",
        values="rmse_bp",
        aggfunc="first",
    ).loc[list(MATURITIES)]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Ablation Test RMSE by Maturity")
    ax.set_xlabel("Maturity")
    ax.set_ylabel("RMSE (basis points)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Model", frameon=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_loss_figure(histories: dict, figure_path: Path) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model_name, history in histories.items():
        epochs = range(1, len(history["validation_loss"]) + 1)
        ax.plot(epochs, history["validation_loss"], marker="o", label=model_name)
    ax.set_title("Validation Loss During Ablation Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_summary(metrics: pd.DataFrame, histories: dict) -> dict:
    overall_test = metrics[
        (metrics["split"] == "test") & (metrics["maturity"] == "overall")
    ].copy()
    by_model = {
        row["model"]: {
            "mae_bp": float(row["mae_bp"]),
            "rmse_bp": float(row["rmse_bp"]),
        }
        for _, row in overall_test.iterrows()
    }
    daily = by_model["daily_lstm"]
    multi = by_model["multi_frequency_lstm"]
    if multi["rmse_bp"] < daily["rmse_bp"] and multi["mae_bp"] < daily["mae_bp"]:
        conclusion = "supported"
    elif multi["rmse_bp"] < daily["rmse_bp"] or multi["mae_bp"] < daily["mae_bp"]:
        conclusion = "partially supported"
    else:
        conclusion = "not supported"

    return {
        "hypothesis": (
            "Adding 5-day and 21-day yield-change features improves LSTM "
            "forecast performance over daily-only features."
        ),
        "conclusion": conclusion,
        "overall_test_metrics": by_model,
        "training_histories": histories,
        "note": "Results are reported as run; no result cherry-picking was applied.",
    }
