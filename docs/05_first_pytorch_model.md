# First PyTorch Model

This step adds a deliberately small LSTM. The goal is understanding the training
pipeline, not squeezing out maximum forecast accuracy.

## Core Terms

**Tensor**: a multi-dimensional array. Our input tensor is shaped
`(examples, 60, 28)`, and the target tensor is shaped `(examples, 7)`.
Here, `examples` means how many supervised windows we have, `60` means trading
days per window, `28` means features per day, and `7` means one target for each
Treasury maturity.

**Parameter**: a number inside the model that is learned from data, such as an
LSTM weight or output-layer bias.

**Batch**: a small group of examples processed together. Batches make training
more efficient than processing one example at a time.

**Dataset**: a PyTorch object that returns one `(X, y)` pair by index.

**DataLoader**: an iterator that groups Dataset examples into batches.

**Forward pass**: the calculation that turns input tensors into predictions.

**Loss**: a number measuring prediction error. This model uses mean squared
error.

**Gradient**: the direction each parameter should move to reduce the loss.

**Backpropagation**: the algorithm that computes gradients from the loss back
through the model.

**Optimizer**: the rule that updates parameters using gradients. This model uses
Adam.

**Adam**: an optimizer that adapts the learning rate for each parameter. It is a
common first choice because it is stable and usually works well without much
tuning.

**Early stopping**: a training rule that stops when validation loss stops
improving. It helps prevent the model from memorizing training data after it has
stopped getting better on validation data.

**Epoch**: one full pass through the training examples.

## Tensor Shapes

The model is intentionally small:

- `input_size = 28`
- `hidden_size = 64`
- `num_layers = 1`
- `output_size = 7`

For a batch of 64 examples, the tensor shapes are:

| Step | Tensor shape | Meaning |
|---|---:|---|
| Input batch | `(64, 60, 28)` | 64 examples, 60 trading days, 28 features per day |
| LSTM sequence output | `(64, 60, 64)` | one 64-value hidden vector for every day |
| LSTM hidden state | `(1, 64, 64)` | 1 layer, 64 examples, 64 hidden values |
| LSTM cell state | `(1, 64, 64)` | LSTM memory state with the same layer/batch/hidden shape |
| Final hidden state | `(64, 64)` | last day's hidden vector for each example |
| Linear output | `(64, 7)` | seven predicted yield changes per example |

Those dimensions mean:

- `64` examples in the batch
- `60` trading days in each sequence
- `28` input features per day
- `64` learned hidden values inside the LSTM
- `7` output predictions, one for each maturity

## Training Loop

The training loop includes:

- deterministic seed
- training loss
- validation loss
- MSE loss
- Adam optimizer
- early stopping
- best-model checkpoint

The reproducible command is:

```bash
PYTHONPATH=src python -m market_resonance.training.train_lstm
```

It writes the best checkpoint to `reports/models/first_lstm.pt` by default.
