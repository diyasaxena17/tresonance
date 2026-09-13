# Reinforcement Learning Extension

Tresonance began as a supervised forecasting project: given a window of
Treasury yield history, predict the next-day numerical change for each maturity.
The reinforcement-learning extension asks a smaller question. Instead of
forecasting the whole yield curve, can a small agent learn a useful next-day
direction decision for the 10Y Treasury yield?

This is an educational experiment, not a trading strategy. It keeps the RL
setup deliberately simple so the mechanics are visible.

## Why Add RL?

The original models answer a regression question: "how many basis points will
the yield move?" In practice, a weaker but still useful research question is
directional: "will the yield fall, stay roughly flat, or rise?"

Reinforcement learning is a natural way to express this as a sequential decision
problem. At each date, the agent observes only information available at that
date, chooses one action, and receives a reward after the next trading day is
known.

## Research Question

Can a small Deep Q-Network improve next-day 10Y Treasury direction decisions
relative to the existing Tresonance forecasting approaches when every model is
evaluated on the same held-out dates?

The shared direction classes are:

- `0 = FALL`
- `1 = FLAT`
- `2 = RISE`

The direction threshold is 1 basis point. A next-day 10Y change below `-1 bp`
is FALL, a change between `-1 bp` and `+1 bp` inclusive is FLAT, and a change
above `+1 bp` is RISE.

## Why Only The 10Y Treasury?

The 10Y Treasury yield is a common benchmark for U.S. rates, mortgages, equity
discount rates, and macro-financial conditions. Restricting the first RL
experiment to one maturity keeps the action space and evaluation easy to reason
about.

The supervised models still forecast all seven maturities, but the directional
comparison uses only their 10Y predictions.

## State Representation

The environment state uses current and trailing information only:

- current 10Y yield level
- 1-day 10Y yield change
- 5-day 10Y yield change
- 21-day 10Y yield change
- 2Y-10Y curve spread

These features reuse the project's existing Treasury feature conventions where
possible. The spread is computed as `10Y - 2Y`.

## Actions And Reward

The DQN chooses one of three actions:

| action | meaning |
|---:|---|
| 0 | FALL |
| 1 | FLAT |
| 2 | RISE |

The reward is intentionally sparse:

- `+1` if the chosen action matches the realized next-day direction
- `-1` otherwise

This makes directional accuracy and average directional reward closely related.
Average reward is stricter to read emotionally: negative values mean wrong
direction decisions still outnumber correct ones.

## DQN Architecture

The DQN is a small feed-forward PyTorch network:

```text
Linear(input_dim, 32)
ReLU
Linear(32, 32)
ReLU
Linear(32, 3)
```

The input dimension is inferred from the environment state size. The three
outputs are Q-values for FALL, FLAT, and RISE.

## Replay Buffer And Target Network

The training loop uses standard DQN ingredients:

- an online Q-network that is optimized with Adam
- a target Q-network that is periodically copied from the online network
- a replay buffer that stores training-period transitions
- epsilon-greedy exploration during training
- Bellman targets with discount factor `gamma`
- Smooth L1 loss

Replay sampling is random, but only within transitions collected from the
chronological training period.

## Chronology And Leakage Prevention

The raw Treasury dataset is split chronologically. The environment used for DQN
training is created only from the training slice. No validation or test
transitions enter the replay buffer.

At time `t`, the state contains only current and trailing values. The realized
label is computed from the move between `t` and `t + 1`, after the action is
chosen. This keeps the future row out of the current state.

For the final comparison, all models are evaluated on the same held-out 10Y
target dates. Where artifacts do not cover identical date sets, the evaluator
uses the common intersection.

## Fair Comparison

The compared approaches are:

- zero-change persistence
- linear regression
- daily-only LSTM
- multi-frequency LSTM
- DQN

The supervised models predict numerical next-day 10Y yield changes. Those
predictions are converted into directions with the same 1 bp threshold used by
the RL environment.

The DQN does not produce a numerical yield-change forecast. It chooses a
directional action directly. Therefore MAE and RMSE are not assigned to the DQN.
Directional accuracy and average directional reward are the shared comparison
metrics.

## Actual Results

Results are from `reports/tables/rl_model_comparison.csv` and
`results/rl_metrics.json`.

The common held-out evaluation window contains 1,365 observations from
2021-03-25 through 2026-09-08. The threshold is 1 basis point.

| model | 10Y MAE (bp) | 10Y RMSE (bp) | direction accuracy | average reward |
|---|---:|---:|---:|---:|
| persistence zero-change | 4.708 | 6.048 | 0.136 | -0.727 |
| linear regression | 4.926 | 6.368 | 0.337 | -0.326 |
| daily LSTM | 4.732 | 6.069 | 0.201 | -0.597 |
| multi-frequency LSTM | 4.816 | 6.250 | 0.264 | -0.471 |
| DQN | N/A | N/A | 0.448 | -0.103 |

Class-wise DQN recall:

| class | recall |
|---|---:|
| FALL | 0.606 |
| FLAT | 0.000 |
| RISE | 0.439 |

DQN training episode rewards were:

```text
[-1867.0, -1695.0, -1525.0, -1173.0, -1069.0]
```

## Interpretation

In this run, the DQN had the highest directional accuracy among the compared
models. It beat persistence, linear regression, the daily LSTM, and the
multi-frequency LSTM on the shared directional metric.

That said, the result is not a clean victory. The DQN's average directional
reward remained negative, meaning incorrect actions still outnumbered correct
ones. It also had zero FLAT recall, so it did not learn a balanced three-class
policy. The model appears to have improved the directional decision relative to
the numerical forecasting models, but it did not solve the next-day Treasury
direction problem.

The experiment provides mild evidence that the directional RL formulation is
worth studying further. It does not provide evidence of a profitable strategy or
a production-ready investment system.

## Limitations

- This is a small educational DQN with minimal tuning.
- Only the 10Y maturity is modeled as an RL decision.
- The reward ignores transaction costs, risk, position sizing, and economic
  utility.
- Directional accuracy is not the same as trading profitability.
- The DQN does not provide calibrated numerical forecasts.
- The FLAT class was not handled well in this run.
- The comparison depends on regenerated per-date predictions from existing
  checkpoints because earlier saved result files stored aggregate metrics, not
  dated predictions.
- The result should be checked across additional seeds, thresholds, and market
  regimes before drawing stronger conclusions.
