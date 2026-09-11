# Multi-Frequency Ablation Study

An ablation study tests whether a specific ingredient helps by removing or
changing that ingredient while keeping everything else as identical as practical.

Here the ingredient is multi-frequency feature information:

- Daily model: yield levels plus 1-day changes
- Multi-frequency model: yield levels plus 1-day, 5-day, and 21-day changes

Both models use the same target, lookback length, hidden size, optimizer,
learning rate, early-stopping rule, split method, and seed.

## Hypothesis

Adding 5-day and 21-day yield-change features should improve LSTM forecast
performance over daily-only features.

## Comparisons

The study reports:

- validation MAE and RMSE
- test MAE and RMSE
- error by maturity
- performance relative to persistence
- performance relative to linear regression

The result is recorded as supported, partially supported, or not supported based
on the test-set overall MAE and RMSE. Results are reported as run; no
cherry-picking is applied.

## Result From This Run

The hypothesis was **not supported**.

| model | test MAE (bp) | test RMSE (bp) |
|---|---:|---:|
| persistence zero-change | 3.679 | 5.457 |
| daily LSTM | 3.756 | 5.502 |
| multi-frequency LSTM | 3.805 | 5.544 |
| linear regression | 3.994 | 5.679 |

The daily-only LSTM slightly outperformed the multi-frequency LSTM on overall
test MAE and RMSE. Neither LSTM beat the persistence baseline in this run. That
does not prove multi-frequency features can never help, but this matched
experiment does not support the hypothesis.

## Command

```bash
PYTHONPATH=src python -m market_resonance.evaluation.run_ablation
```

Outputs:

- `reports/tables/ablation_results.csv`
- `reports/figures/ablation_rmse_by_maturity.png`
- `reports/figures/ablation_validation_loss.png`
- `results/ablation_summary.json`
