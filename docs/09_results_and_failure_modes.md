# Results and Failure Modes

All numbers here are pulled from generated result files.

## Baselines

From `reports/tables/baseline_results.csv`:

| model | MAE (bp) | RMSE (bp) |
|---|---:|---:|
| zero-change | 3.679 | 5.457 |
| linear regression | 3.994 | 5.679 |

The zero-change baseline beat the linear regression baseline overall.

## LSTM Ablation

From `results/ablation_summary.json`:

| model | MAE (bp) | RMSE (bp) |
|---|---:|---:|
| persistence zero-change | 3.679 | 5.457 |
| daily LSTM | 3.756 | 5.502 |
| multi-frequency LSTM | 3.805 | 5.544 |
| linear regression | 3.994 | 5.679 |

The multi-frequency hypothesis was **not supported**. The daily-only LSTM
slightly outperformed the multi-frequency LSTM, and neither LSTM beat
persistence.

## Inference

From `results/inference_metrics.json`:

- parameter count: 24,519
- single-sample latency: 0.5181 ms
- batch latency for 64 samples: 7.5259 ms

## Uncertainty

From `reports/tables/uncertainty_ranges.csv`, the 90% Monte Carlo range widths
increase from shorter to longer maturities in this run:

- 3M: 8.157107 bp
- 2Y: 11.536339 bp
- 30Y: 14.869325 bp

## Failure Modes

The core modeling failure is that the neural models did not beat persistence.
Daily Treasury changes are noisy, often small, and hard to predict. A model can
learn patterns in training data without improving out-of-sample forecasts.

Other limitations:

- The LSTM is intentionally small and lightly tuned.
- The target horizon is only one trading day.
- Constant-maturity/par yields are not a zero-coupon discount curve.
- The stochastic simulation assumes validation residuals are representative and
approximately multivariate normal.
- The uncertainty layer models forecast-error covariance, not all market risks.
