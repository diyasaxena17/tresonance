# Market Resonance

**Multi-Frequency Neural Yield Curve Forecasting with PyTorch**

Market Resonance is a research-style project asking whether daily, weekly, and monthly representations of U.S. Treasury yield movements improve out-of-sample forecasts over simple baselines.

This repository is intentionally being built in small conceptual stages. The model is not implemented yet.

## Research scope

We will forecast seven Treasury maturities:

`3M`, `6M`, `1Y`, `2Y`, `5Y`, `10Y`, and `30Y`.

The primary model will be a small, interpretable PyTorch LSTM. Every comparison will preserve chronological order and keep preprocessing statistics fitted on training data only.

## Setup

The project currently targets Python 3.9 or newer. A fresh environment can be created with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Then verify the data phase:

```bash
pytest
ruff check .
```

## Download Treasury data

After installing the project, download the complete daily observations from
FRED with:

```bash
download-treasury-data --start-date 1990-01-01
```

This writes `data/processed/treasury_yields_daily.csv`. The command uses the
public FRED CSV endpoint and does not require an API key. It keeps one row per
date only when all seven maturities have numeric observations. It does not
forward-fill weekends, holidays, or other missing observations.

## Project map

See [docs/architecture.md](docs/architecture.md) for the planned stages, tensor contracts, leakage controls, and experiment flow.

- `src/market_resonance/`: package code, added one conceptual stage at a time
- `tests/`: small tests for data and feature logic
- `data/raw/`: downloaded source data, never committed
- `data/interim/`: cleaned intermediate data, never committed
- `data/processed/`: model-ready artifacts, never committed
- `reports/figures/`: generated publication-quality figures, never committed
- `configs/`: version-controlled experiment settings
- `notebooks/`: optional exploration only; reusable logic belongs in `src/`
