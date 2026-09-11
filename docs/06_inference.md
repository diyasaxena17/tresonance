# Inference

Training and inference are separate workflows.

Training is where the model learns. It uses input tensors, target labels, loss,
gradients, backpropagation, an optimizer, epochs, validation loss, early
stopping, and checkpoint saving.

Inference is where a saved model is used. It does **not** learn new parameters.
It loads the checkpoint, loads the training normalization statistics, prepares
the latest 60-day input sequence, runs the model once or in batches, and returns
a forecast.

## What Exists During Training But Not Inference

These are used during training:

- target labels `y`
- loss function
- gradients
- backpropagation
- optimizer steps
- epochs
- validation-loss monitoring
- early stopping
- train DataLoader shuffling

These are not used during inference:

- no target labels are needed
- no loss is computed
- no gradients are computed
- no backpropagation happens
- no optimizer updates happen
- no epochs happen
- no parameters are changed

The inference command explicitly calls `model.eval()` and runs inside
`torch.no_grad()` so PyTorch knows this is prediction-only work.

## Forecast Output

The model predicts seven future yield changes, one for each maturity. Targets
were trained as yield changes in percentage points, so inference converts them
to:

- predicted change in basis points
- forecast yield level in percent

The command also measures:

- trainable parameter count
- single-sample inference latency
- batch inference latency

## Command

```bash
PYTHONPATH=src python -m market_resonance.inference.run_inference
```

By default this writes:

`results/inference_metrics.json`
