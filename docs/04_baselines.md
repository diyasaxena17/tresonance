# Phase 4 Baselines

Baselines are the first models in the forecasting stack. They are intentionally
simple, transparent, and hard to beat for noisy financial data.

## Why Baselines Matter

Baselines are critical in ML research because they create a reality check. A
complex model is not useful just because it is complex. It must beat simple
alternatives on the same data, same split, and same metrics.

The zero-change baseline is especially important for daily Treasury changes
because many daily moves are small. It predicts that every maturity's next
change is zero. This is equivalent to saying tomorrow's yield level is the same
as today's.

The linear regression baseline asks whether a simple weighted combination of the
60-day feature history can forecast next-day changes. It is still interpretable
and much simpler than a neural network.

## Evaluation Metrics

Errors are reported in basis points:

- `MAE`: average absolute forecast error
- `RMSE`: square-root average squared forecast error
- error by maturity: the same metrics separately for 3M through 30Y

RMSE penalizes large misses more heavily than MAE.

## Outputs

The reproducible command is:

```bash
PYTHONPATH=src python -m market_resonance.evaluation.run_baselines
```

It writes:

- `reports/tables/baseline_results.csv`
- `reports/figures/baseline_rmse_by_maturity.png`

## Current Overall Test Results

| model | MAE (bp) | RMSE (bp) |
|---|---:|---:|
| zero-change | 3.679 | 5.457 |
| linear regression | 3.994 | 5.679 |

On this test split, the zero-change baseline is slightly stronger overall than
the linear regression baseline. That is a useful warning: daily yield changes
are noisy, and a more complex model must beat a very simple "no change"
forecast before it is worth trusting.

No LSTM or neural network is implemented in this phase.
