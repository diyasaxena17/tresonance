# Training

Training uses the supervised windows from the feature pipeline.

## Core Terms

**Tensor**: a multi-dimensional array. Our input tensor is shaped
`(examples, 60, 28)`, and the target tensor is shaped `(examples, 7)`. Here,
`examples` means supervised windows, `60` means trading days per window, `28`
means features per day, and `7` means one target for each Treasury maturity.

**Parameter**: a number inside the model that is learned from data, such as an
LSTM weight or output-layer bias.

**Batch**: a small group of examples processed together.

**Dataset**: a PyTorch object that returns one `(X, y)` pair by index.

**DataLoader**: an iterator that groups Dataset examples into batches.

**Forward pass**: the calculation that turns input tensors into predictions.

**Loss**: a number measuring prediction error. This model uses mean squared
error.

**Gradient**: the direction each parameter should move to reduce the loss.

**Backpropagation**: the algorithm that computes gradients from the loss back
through the model.

**Optimizer**: the rule that updates parameters using gradients. This model
uses Adam.

**Adam**: an optimizer that adapts the learning rate for each parameter.

**Early stopping**: a rule that stops training when validation loss stops
improving.

**Epoch**: one full pass through the training examples.

## Components

Training includes:

- Dataset and DataLoader
- batches
- forward pass
- MSE loss
- gradients
- backpropagation
- Adam optimizer
- validation loss
- early stopping
- best-model checkpoint

## Tensor Shapes

For a batch of 64 multi-frequency examples:

| Step | Tensor shape | Meaning |
|---|---:|---|
| Input batch | `(64, 60, 28)` | 64 examples, 60 trading days, 28 features per day |
| LSTM sequence output | `(64, 60, 64)` | one 64-value hidden vector for every day |
| LSTM hidden state | `(1, 64, 64)` | 1 layer, 64 examples, 64 hidden values |
| LSTM cell state | `(1, 64, 64)` | LSTM memory state |
| Final hidden state | `(64, 64)` | last day's hidden vector for each example |
| Linear output | `(64, 7)` | seven predicted yield changes per example |

## Settings

The deliberately simple default LSTM settings are:

- hidden size: 64
- one LSTM layer
- MSE loss
- Adam optimizer
- deterministic seed: 42

The ablation study trains two matched LSTMs:

- daily-only LSTM
- multi-frequency LSTM

## Checkpoints

Training checkpoints include:

- model weights
- model configuration
- feature columns
- target columns
- lookback and horizon
- training normalization mean and scale

The extra metadata is required so inference can normalize new input sequences in
the same way training did.
