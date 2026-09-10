# Machine-Learning Terminology

This project turns Treasury yield data into examples a model can learn from.
These terms explain the language used in the feature-window code and notebooks.

## Supervised learning

Supervised learning means each training example has an input and a correct
answer. In this project, the input is recent Treasury yield history, and the
answer is the future change in yields.

## Input features

Features are the numbers given to the model. For each maturity, we use the yield
level plus recent trailing changes over 1, 5, and 21 trading days.

## Target labels

The target label is what the model is asked to predict. Here, the label is a
vector of future yield changes for all seven maturities. With a 1-day horizon,
the label is tomorrow's change from today's yield level.

## Lookback window

The lookback window is how much history one example sees. A 60-trading-day
lookback means each `X` example contains the previous 60 trading rows ending at
the forecast origin date.

## Forecast horizon

The horizon is how far into the future the target looks. A horizon of 1 predicts
the next trading day. A horizon of 5 predicts the change five trading rows ahead.

## Batch-first tensors

A tensor is a multi-dimensional array. Batch-first means the first dimension
counts independent examples. Our input tensor shape is:

`(num_examples, 60, 28)`

The dimensions mean:

- `num_examples`: how many supervised examples are in the split
- `60`: trading days in each input sequence
- `28`: features per day

The target tensor shape is:

`(num_examples, 7)`

That means one future-change target for each of the seven maturities.

## Normalization

Normalization rescales features so they are easier for a model to use. This
project uses standardization: subtract the mean and divide by the standard
deviation.

The important rule is that normalization parameters are fitted using training
data only. Validation and test data must not influence the mean or scale,
because that would leak future information into the training process.

## Chronological split

A chronological split keeps time order intact. The oldest examples become the
training set, the next block becomes validation, and the newest block becomes
test. We do not randomly shuffle time-series examples before splitting.

## Leakage

Leakage happens when information from the future accidentally enters the model's
inputs or preprocessing. Leakage can make results look better than they really
are. The Phase 3 tests check that future rows do not enter `X` and that
normalization is fitted only on training inputs.
