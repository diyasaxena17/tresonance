# Supervised Feature Windows

This step turns the cleaned Treasury yield table into supervised-learning
examples. The raw data is a time series. A supervised dataset has inputs and
targets:

- `X`: information known up to a forecast origin date `t`
- `y`: what happens after `t`

For each maturity, the input features are:

- yield level
- 1-trading-day change
- 5-trading-day change
- 21-trading-day change

With seven maturities, this gives `7 * 4 = 28` features at each time step.

## One Example

With a 60-trading-day lookback and a 1-trading-day horizon:

- `X` is the previous 60 trading days ending at date `t`
- `y` is the next day's seven yield changes, from `t` to `t + 1`

So one example has:

- `X_i.shape == (60, 28)`
- `y_i.shape == (7,)`

The full dataset uses batch-first tensors:

- `X.shape == (num_examples, 60, 28)`
- `y.shape == (num_examples, 7)`

The horizon is configurable. If `horizon=5`, the target becomes the change from
`t` to `t + 5` trading rows.

## Leakage Controls

Feature rows use only current or past yields. Future rows are used only to build
the target labels. Train, validation, and test examples are split
chronologically. Normalization is fitted on training inputs only, then reused for
validation and test.
