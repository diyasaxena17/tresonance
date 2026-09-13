# Abstract

Tresonance studies whether multi-frequency Treasury yield features improve
short-horizon yield-curve forecasts. The project uses daily U.S. Treasury
constant-maturity yields for seven maturities: 3M, 6M, 1Y, 2Y, 5Y, 10Y, and
30Y.

The pipeline downloads public FRED data without an API key, creates supervised
60-trading-day windows, compares simple baselines with small PyTorch LSTMs, and
adds a simple stochastic uncertainty layer using validation residuals.

The central hypothesis was that adding 5-day and 21-day changes to daily yield
features would improve neural forecasts. In the generated ablation run, this
hypothesis was **not supported**: the daily-only LSTM slightly outperformed the
multi-frequency LSTM, and neither LSTM beat the persistence baseline.

The strongest result is methodological rather than predictive: the repository
demonstrates a leakage-aware research pipeline with chronological splits,
training-only normalization, reproducible baselines, model checkpoints,
inference separation, and documented failure modes.
