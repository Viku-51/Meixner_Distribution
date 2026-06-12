# Implementation Documentation: Meixner Distribution for Equity Risk Modeling

## 1. Overview

This document describes the mathematical framework, code architecture,
and statistical validation results for the Meixner distribution
implementation used in equity risk modeling. It is intended as a
self-contained technical reference accompanying the source code in
`src/`.

---

## 2. Mathematical Framework

### 2.1 The Meixner Distribution

The Meixner distribution belongs to the class of generalized hyperbolic
distributions and has the probability density function:

```
f(x; α, β, δ, m) = ((2 cos(β/2))^(2δ)) / (2 α π Γ(2δ)) · exp(β(x − m)/α) · |Γ(δ + i(x − m)/α)|²
```

**Parameters:**

- `α > 0` — scale parameter
- `−π < β < π` — skewness/asymmetry parameter
- `δ > 0` — shape parameter (controls tail thickness / kurtosis)
- `m ∈ ℝ` — location parameter

**Special functions involved:** The term `Γ(δ + iz)` is the complex gamma
function evaluated at a complex argument. The implementation evaluates
`|Γ(δ + iz)|² = exp(2 · Re[log Γ(δ + iz)])` using `scipy.special.loggamma`,
which is numerically stable for large arguments (avoids overflow that a
direct `gamma()` call would produce).

### 2.2 Theoretical Moments

The Meixner distribution has closed-form moments in terms of its
parameters:

| Moment | Formula |
|---|---|
| Mean | `m + αδ·tan(β/2)` |
| Variance | `α²δ / cos²(β/2)` |
| Skewness | `sin(β/2)·√(2/δ)` |
| Excess Kurtosis | `(2 − cos(β)) / δ` |

These relationships are used in two places:

1. **Starting values for MLE** — sample moments (mean, variance, skewness,
   excess kurtosis) are inverted numerically (via `scipy.optimize.brentq`
   on the kurtosis/skewness relationship for `β`) to produce a good initial
   guess for the optimizer.
2. **Diagnostics** — after fitting, the theoretical moments implied by the
   fitted parameters are compared against the empirical sample moments as
   a sanity check.

### 2.3 Maximum Likelihood Estimation

Given a sample of `n` daily returns `x_1, ..., x_n`, the MLE estimates
maximize the log-likelihood:

```
ℓ(α, β, δ, m) = Σ log f(x_i; α, β, δ, m)
```

This is implemented as a numerical minimization of the negative
log-likelihood using `scipy.optimize.minimize` with the **L-BFGS-B**
method, subject to the parameter constraints `α > 0`, `δ > 0`,
`−π < β < π` (enforced via `bounds`).

**Convergence diagnostics returned:**

- Final log-likelihood
- AIC = `2k − 2ℓ` (k = 4 parameters)
- BIC = `k·ln(n) − 2ℓ`
- Optimizer convergence flag and message

---

## 3. Value-at-Risk (VaR) Estimation

### 3.1 Definition

For a confidence level `c` (e.g., 99%), the Value-at-Risk is defined as
the magnitude of loss such that:

```
P(R < −VaR) = 1 − c
```

i.e., VaR is the negative of the `(1−c)`-quantile of the return
distribution `R`.

### 3.2 Numerical Quantile Computation

Because the Meixner CDF has no closed form, the `(1−c)`-quantile is found
by:

1. Establishing a numerical search bracket using the theoretical mean ±
   20 standard deviations (derived from the fitted parameters).
2. Numerically integrating the PDF from the lower bracket bound to a
   trial point `x` using `scipy.integrate.quad` to obtain `CDF(x)`.
3. Using `scipy.optimize.brentq` to solve `CDF(x) = 1 − c` for `x`.

This bounded, numerically robust approach avoids the overflow issues that
can occur with `scipy.stats.rv_continuous.ppf` for distributions with
complex-valued density kernels.

### 3.3 Rolling-Window Backtest Pipeline

`rolling_meixner_var()` implements a realistic production-style VaR
pipeline:

- A rolling estimation window of `W` observations (default 250 trading
  days, ≈ 1 year).
- The Meixner distribution is re-fit via MLE every `refit_every`
  observations (default 20 days) — re-fitting daily would be
  computationally wasteful and is not how most production systems operate.
- For each day `t`, the VaR estimate uses the most recently fitted
  parameters.

A parallel `normal_var()` function computes the standard parametric Normal
VaR (`-(μ + z·σ)`, with `z` the Normal quantile at `1−c`) using the same
rolling window, for benchmark comparison.

---

## 4. Backtesting Framework

### 4.1 Kupiec Proportion of Failures (POF) Test

The Kupiec (1995) test evaluates whether the observed number of VaR
exceptions `x` out of `n` observations is consistent with the expected
exception probability `p = 1 − c` under the null hypothesis. The
likelihood-ratio statistic is:

```
LR_POF = −2 · ln[ (1−p)^(n−x) p^x / (1−x/n)^(n−x) (x/n)^x ]
```

Under H₀, `LR_POF ~ χ²(1)`. A p-value below 0.05 leads to rejection of the
model at the 95% significance level — i.e., the VaR model is statistically
mis-calibrated (either too conservative or too liberal).

**Edge cases handled:**

- `x = 0` (no exceptions) and `x = n` (all exceptions) are handled with
  limiting-case log-likelihoods to avoid `log(0)`.
- The LR statistic is floored at 0 to guard against tiny negative values
  from floating-point error.

### 4.2 Basel Traffic Light Approach

As a complementary, regulator-style check, `kupiec_traffic_light()`
classifies the number of exceptions in a 250-day window into:

- **Green** (0–4 exceptions at 99% confidence): model performing as
  expected.
- **Yellow** (5–9 exceptions): possible model deterioration; increased
  capital multiplier under Basel rules.
- **Red** (10+ exceptions): the model significantly underestimates risk.

For non-standard window sizes / confidence levels, the function falls back
to a cumulative-binomial-probability classification.

---

## 5. Validation Results Summary

On the synthetic 2,000-day daily return series included in `data/`
(empirical skewness ≈ −0.44, excess kurtosis ≈ 3.17):

**Fitted Meixner parameters (full sample MLE):**

| Parameter | Value |
|---|---|
| α | 0.0245 |
| β | −0.2568 |
| δ | 0.2867 |
| m | 0.000538 |
| Log-likelihood | 6649.6 |
| AIC | −13291.2 |
| BIC | −13268.8 |

**Implied theoretical moments vs. empirical:**

| Moment | Empirical | Meixner-implied |
|---|---|---|
| Mean | −0.000368 | −0.000370 |
| Skewness | −0.4439 | −0.3382 |
| Excess Kurtosis | 3.1672 | 3.6021 |

**99% VaR Kupiec POF backtest (1,750 out-of-sample observations):**

| Model | Exceptions | Breach Rate | LR Statistic | p-value | Decision |
|---|---|---|---|---|---|
| Meixner | 25 | 1.43% | 2.87 | 0.090 | Accept |
| Normal | 37 | 2.11% | 16.63 | <0.0001 | Reject |

**Basel Traffic Light (last 250 days):**

| Model | Exceptions | Zone |
|---|---|---|
| Meixner | 2 | Green |
| Normal | 4 | Green |

**Interpretation:** Both models are "Green" in the most recent 250-day
window, but over the full 1,750-day out-of-sample period the Normal
model's exception rate (2.11%) is more than double the target 1%, and is
statistically rejected by the Kupiec test. The Meixner model's exception
rate (1.43%) remains statistically consistent with the 99% confidence
target — demonstrating the practical benefit of modeling heavy tails and
skewness explicitly rather than assuming Normality.

---

## 6. Code Quality & Testing

The implementation includes 16 unit tests (`tests/test_meixner.py`)
covering:

- PDF positivity and normalization (integrates to 1).
- Consistency between `pdf` and `logpdf`.
- Theoretical moment relationships (symmetric case has zero skew; sign of
  `β` controls sign of skewness).
- MLE convergence on simulated heavy-tailed and skewed data.
- VaR monotonicity with respect to confidence level.
- Kupiec POF acceptance/rejection on correctly- and incorrectly-calibrated
  VaR series.
- Basel Traffic Light zone classification.

All 16 tests pass.

---

## 7. Limitations & Future Work

- **Static window assumption:** The rolling backtest assumes the Meixner
  parameters are constant between refits (every 20 days). A fully dynamic
  (e.g., time-varying `δ` linked to a GARCH volatility process) model could
  improve responsiveness to regime changes.
- **Univariate only:** The current implementation models a single return
  series. Portfolio-level VaR would require a multivariate extension
  (e.g., Meixner Lévy processes with dependence structures).
- **Synthetic data:** Results presented here use a synthetic GARCH-with-jumps
  return series. Real market data (equities, FX, rates) should be
  substituted for production validation, as described in the README.
