"""Tests for the multi-frequency ablation reporting helpers."""

import pandas as pd

from market_resonance.evaluation.ablation import _build_summary


def test_ablation_summary_marks_supported_when_multi_frequency_wins() -> None:
    """The hypothesis is supported only when multi-frequency improves both metrics."""
    metrics = pd.DataFrame(
        {
            "split": ["test", "test"],
            "feature_set": ["daily_lstm", "multi_frequency_lstm"],
            "model": ["daily_lstm", "multi_frequency_lstm"],
            "maturity": ["overall", "overall"],
            "mae_bp": [5.0, 4.0],
            "rmse_bp": [6.0, 5.0],
        }
    )

    summary = _build_summary(metrics, histories={})

    assert summary["conclusion"] == "supported"


def test_ablation_summary_marks_not_supported_when_daily_wins() -> None:
    """The hypothesis is not supported when daily-only is better on both metrics."""
    metrics = pd.DataFrame(
        {
            "split": ["test", "test"],
            "feature_set": ["daily_lstm", "multi_frequency_lstm"],
            "model": ["daily_lstm", "multi_frequency_lstm"],
            "maturity": ["overall", "overall"],
            "mae_bp": [4.0, 5.0],
            "rmse_bp": [5.0, 6.0],
        }
    )

    summary = _build_summary(metrics, histories={})

    assert summary["conclusion"] == "not supported"
