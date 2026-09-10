# Architecture and Research Plan

## Research question

Do daily, weekly, and monthly representations of U.S. Treasury yield movements improve out-of-sample yield-curve forecasts over simple baselines?

The seven target maturities are `3M`, `6M`, `1Y`, `2Y`, `5Y`, `10Y`, and `30Y`. The project will compare models using the same dates, target definition, and evaluation windows.

## Planned stages

1. **Data contract and download**: use the public FRED CSV endpoint for DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10, and DGS30. The cleaned artifact stores dates plus the seven yields in percent, with incomplete dates removed and no forward-fill.
2. **Cleaning and inspection**: parse dates, sort chronologically, detect duplicates and missing observations, and save an immutable cleaned intermediate artifact.
3. **Feature construction**: create daily, weekly, and monthly movement representations without using observations after the prediction time.
4. **Chronological splits**: create train, validation, and test periods. The test period remains untouched until final evaluation.
5. **Preprocessing**: fit transformations such as standardization on training rows only, then apply the frozen parameters to validation and test rows.
6. **Baselines**: establish persistence and simple statistical benchmarks before introducing neural networks.
7. **Small LSTM**: train the primary model on fixed-length historical windows.
8. **Ablations**: compare daily-only, weekly-only, monthly-only, and combined-frequency inputs.
9. **Evaluation and reporting**: save actual metrics, forecast tables, residual diagnostics, and publication-quality figures under `reports/figures/`.

No model or data implementation belongs in the scaffold phase.

## Planned tensor contract

The exact feature count will be finalized during feature design. The core batch-first sequence convention is:

- Input tensor: `(batch_size, lookback_steps, num_features)`
  - axis 0: independent historical examples in one training batch
  - axis 1: ordered time steps in the lookback window
  - axis 2: features available at each time step
- Target tensor: `(batch_size, 7)`
  - axis 0: examples aligned with the input windows
  - axis 1: the seven forecast maturities in the fixed order above
- LSTM output: `(batch_size, lookback_steps, hidden_size)` when `batch_first=True`
  - axis 0: examples
  - axis 1: time steps
  - axis 2: learned hidden representation
- Selected final hidden representation: `(batch_size, hidden_size)`
- Prediction tensor: `(batch_size, 7)`
  - one forecast per maturity for each example

These shapes will be asserted in tests when the data-window and model stages are implemented.

## Leakage controls

- Chronological splits are never randomly shuffled.
- Any scaler, imputer, feature-selection rule, or learned statistic is fitted using training data only.
- Validation is used for model and hyperparameter decisions; it is not reported as final performance.
- Test data is read only for the final locked evaluation.
- Features at forecast origin `t` may use information available by `t`, but never `t + 1` or later.
- Raw downloads and generated reports are excluded from Git.
- Seeds will be fixed and recorded for Python, NumPy, and PyTorch runs.

## Experiment layout

Each experiment should record its configuration, seed, data version or source metadata, split dates, preprocessing parameters, training history, and final metrics. A future run directory can live under `reports/runs/` without changing source code.

Primary metrics will be selected after the target horizon is defined. Candidate metrics are MAE, RMSE, and per-maturity errors, with aggregate values reported alongside the full maturity breakdown. No result will be described until it comes from an actual run.

## Ownership boundaries

- `data/`: artifacts and provenance, not business logic
- `src/market_resonance/data/`: loading, validation, cleaning, and splitting
- `src/market_resonance/features/`: frequency representations and window creation
- `src/market_resonance/models/`: baselines and the small LSTM
- `src/market_resonance/training/`: training loop, early stopping, and seed handling
- `src/market_resonance/evaluation/`: metrics and forecast comparisons
- `src/market_resonance/visualization/`: figures and tables
- `tests/`: focused tests for each data or feature contract
