# PyTorch and Model Architecture

The neural model is intentionally small so the mechanics are easy to inspect.

## Baseline Context

Before using the LSTM, the project evaluates simple baselines. Baselines are
critical because a complex model is only useful if it improves on simple,
transparent alternatives under the same data split and metrics.

The zero-change baseline predicts no next-day yield move. The linear regression
baseline uses a weighted combination of the same historical feature window.
Errors are reported in basis points using MAE, RMSE, and per-maturity metrics.

Current overall test results:

| model | MAE (bp) | RMSE (bp) |
|---|---:|---:|
| zero-change | 3.679 | 5.457 |
| linear regression | 3.994 | 5.679 |

On this split, persistence is slightly stronger than linear regression, which
is a useful warning about how noisy daily yield changes are.

## LSTM

The model is a one-layer PyTorch LSTM followed by a linear output layer:

- `input_size`: number of input features
- `hidden_size`: 64
- `num_layers`: 1
- `output_size`: 7

For a batch of 64 multi-frequency examples:

| Step | Shape |
|---|---:|
| Input batch | `(64, 60, 28)` |
| LSTM sequence output | `(64, 60, 64)` |
| Final hidden state | `(64, 64)` |
| Linear output | `(64, 7)` |

## Parameters

The saved inference model reports `24,519` trainable parameters in
`results/inference_metrics.json`.

## Why LSTM?

An LSTM can read a sequence of historical yield-curve features and compress the
last 60 trading days into a hidden state. The model then maps that hidden state
to seven predicted yield changes.

This is not presented as an optimal architecture. It is the first neural model
in a controlled research pipeline.
