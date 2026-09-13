# Research Question

## Question

Do multi-frequency representations of U.S. Treasury yield movements improve
one-trading-day-ahead yield-curve forecasts over simple baselines?

## Hypothesis

Yield changes over multiple horizons may contain useful information that daily
changes alone miss. The project tests whether adding 5-trading-day and
21-trading-day changes improves a small LSTM forecast.

## Forecast Target

The target is the future change in all seven maturities:

`3M`, `6M`, `1Y`, `2Y`, `5Y`, `10Y`, and `30Y`.

The default horizon is one trading day.

## Evaluation

Models are evaluated with:

- MAE in basis points
- RMSE in basis points
- error by maturity
- comparisons against persistence and linear regression

All splits are chronological. Preprocessing is fitted on training data only.
