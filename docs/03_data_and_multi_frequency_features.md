# Data and Multi-Frequency Features

## Data Source

The project uses U.S. Treasury constant-maturity yield series from FRED:

- DGS3MO
- DGS6MO
- DGS1
- DGS2
- DGS5
- DGS10
- DGS30

The cleaned dataset stores one row per date and one yield column per maturity.
Rows with missing maturities are removed rather than forward-filled.

## Features

For each maturity, the feature builder can create:

- yield level
- 1-trading-day change
- 5-trading-day change
- 21-trading-day change

The standard supervised window uses a 60-trading-day lookback.

## Supervised Windows

The cleaned Treasury table is a time series. To train a supervised model, the
pipeline turns it into examples with inputs and labels:

- `X`: information known up to a forecast origin date `t`
- `y`: what happens after `t`

With a 60-trading-day lookback and a 1-trading-day horizon:

- `X` is the previous 60 trading days ending at date `t`
- `y` is the next day's seven yield changes, from `t` to `t + 1`

One example has:

- `X_i.shape == (60, 28)`
- `y_i.shape == (7,)`

If `horizon=5`, the label becomes the change from `t` to `t + 5` trading rows.

The daily-only model uses:

- yield levels
- 1-day changes

The multi-frequency model uses:

- yield levels
- 1-day changes
- 5-day changes
- 21-day changes

## Tensor Contract

For the multi-frequency model:

- `X.shape == (num_examples, 60, 28)`
- `y.shape == (num_examples, 7)`

The 28 features come from 7 maturities times 4 features per maturity.

## Leakage Controls

- Dates are sorted chronologically.
- Splits are chronological, never random.
- Normalization is fitted on training windows only.
- Future observations are used only as labels, never as inputs.
