# Forecasted Rates and Time Value of Money

Interest rates matter because future cash flows are discounted back to the
present.

The basic discounting formula is:

```text
present value = future cash flow / (1 + discount rate) ^ years
```

## Illustrative Approximation

Important: Treasury constant-maturity/par yields are **not** an exact
zero-coupon discount curve. The calculation below is a simplified educational
approximation that treats a maturity yield like a discount rate for a single
cash flow. This is useful for intuition, not bond pricing.

Suppose an investor expects to receive `$1,000` in 10 years.

Using the latest 10Y yield from `results/inference_metrics.json`:

```text
latest 10Y yield = 4.80%
forecast 10Y yield = 4.8091%
```

Approximate present value at 4.80%:

```text
1000 / (1 + 0.0480) ^ 10 = 625.41
```

Approximate present value at 4.8091%:

```text
1000 / (1 + 0.048091) ^ 10 = 624.87
```

In this simplified example, the forecasted 10Y rate is slightly higher, so the
present value is slightly lower. The intuition is general: higher discount rates
reduce present values, while lower discount rates increase present values.

For real Treasury valuation, one would use a properly bootstrapped discount
curve or zero-coupon curve rather than treating constant-maturity/par yields as
exact discount rates.
