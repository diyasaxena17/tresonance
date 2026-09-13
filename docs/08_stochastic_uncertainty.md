# Stochastic Uncertainty

This extension turns a single neural forecast into a simple probabilistic
forecast.

## Terms

**Residual**: the forecast error from a past example. It is actual change minus
predicted change.

**Variance**: how spread out one maturity's forecast errors are.

**Covariance**: how two maturities' forecast errors move together. Positive
covariance means errors tend to be high or low together.

**Covariance matrix**: a table of variances and covariances for all seven
maturities. The diagonal holds variances. The off-diagonal entries hold
covariances between maturities.

**Multivariate normal distribution**: a bell-curve-like probability model for
several variables at once. Here it is used to generate seven related forecast
errors at the same time.

**Point forecast versus probabilistic forecast**: a point forecast gives one
best estimate. A probabilistic forecast gives a range of plausible outcomes
around that estimate.

## Method

The model first produces a neural point forecast for the latest yield curve. We
then estimate the covariance matrix using validation residuals only. Test
residuals are not used.

Next, we sample about 1,000 seven-dimensional error vectors from a multivariate
normal distribution with that covariance matrix. Each sampled error is added to
the neural point forecast to create a simulated future yield curve.

## Assumptions

This is intentionally simple:

- validation residuals are representative of near-future errors
- residuals are centered at zero around the point forecast
- residuals are approximately multivariate normal
- the covariance matrix is stable for this one-step forecast

These assumptions are useful for learning, but they are not a full risk model.

## Outputs

Command:

```bash
PYTHONPATH=src python -m market_resonance.inference.run_uncertainty
```

Outputs:

- `results/monte_carlo_yield_curves.csv`
- `results/uncertainty_summary.json`
- `reports/tables/uncertainty_ranges.csv`
- `reports/figures/yield_curve_fan.png`
- `reports/figures/uncertainty_ranges_by_maturity.png`
