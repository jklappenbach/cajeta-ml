---
id: ml-linear-models
applies-to: [dev.cajeta.ml.LinearRegression, dev.cajeta.ml.Ridge, dev.cajeta.ml.Lasso, dev.cajeta.ml.ElasticNet, dev.cajeta.ml.LogisticRegression, dev.cajeta.ml.SummaryResult]
title: Linear models — OLS with inference, Ridge, Lasso/ElasticNet (cd), Logistic (binary/OvR/softmax)
description: Which linear model for which job, the fitted-surface accessors (coefMatrix vs coef, intercepts), summary() semantics (classical OLS vs penalized Wald), the cd models' exact-sklearn behavior + sparse fitSparse contract, and the multinomial knob.
---

# Linear models

## Pick the model

| Situation | Model | Why |
|---|---|---|
| Baseline fit + you need p-values / R² / F | `LinearRegression(fitIntercept)` | QR lstsq (sklearn's algorithm); `summary()` is classical inference |
| Collinear / ill-conditioned features | `Ridge(alpha, fitIntercept, svdSolver)` | L2 stabilizes; `svdSolver=true` survives near-singular Grams |
| Feature selection / sparse coefficients | `Lasso(alpha)` | soft-threshold → EXACT zeros; `sparsityCount()` |
| Between Ridge and Lasso | `ElasticNet(alpha, l1Ratio)` | `l1Ratio=1` ≡ Lasso bit-for-bit; `→0` approaches Ridge |
| Classification | `LogisticRegression(c, tol, maxIter[, multinomial])` | newton-cholesky IRLS; see below |
| n ≫ nnz (mostly-zero design) | `Lasso/ElasticNet.fitSparse(CsrMatrix, y)` | CSC-mirror cd; bit-identical to dense |

## Fitted surfaces (names differ by model — don't guess)

- `LinearRegression` / `Ridge`: `coefMatrix() (p, k)`, `interceptVector()
  (k,)` — matrix-shaped because both support multi-output `y (n, k)`.
  LinearRegression adds `rank()` and `singularValues()` (numpy ε
  tolerance; rank-deficient designs silently get the min-norm solution —
  the diagnostics are how you see it).
- `Lasso` / `ElasticNet`: `coef() (p,)`, `intercept()` scalar,
  `sparsityCount()`, `iterations()`, `converged()`. **Single target
  only** — `(n, k)` y throws.
- `LogisticRegression`: `coefMatrix() (p, K')`, `interceptVector() (K',)`
  where `K'` = 1 (binary) or K (OvR / multinomial); `predictProba() (n,
  K)`; `predict()` labels; `converged()`.

## `summary(x, y)` — two very different meanings

- **OLS `LinearRegression.summary` = classical inference** (the
  statsmodels table): per-coefficient stderr, t statistics, **exact
  t-distribution p-values** (via `Stats.betainc`), R²/adjusted, F,
  condition number (`condWarning` above 1e8). Index 0 is the intercept.
  Computed with the pseudo-inverse, so collinear designs warn instead of
  exploding.
- **Logistic `summary` (binary fits only) = Wald statistics from the
  PENALIZED Hessian**, and `SummaryResult.penalized == true` says so.
  Don't present these as classical p-values; for near-classical behavior
  pass a very large `C`.

## Lasso / ElasticNet — sklearn's exact algorithm

Cyclic coordinate descent with residual-maintained soft-threshold updates
and the duality-gap stopping rule — the same math as sklearn's cython
solver, so coefficients match the reference and **iteration counts match
exactly**. Objective: `1/(2n)·‖y−Xw‖² + α·l1r·‖w‖₁ + α·(1−l1r)/2·‖w‖²`.
Zeros are exact (a real `== 0.0`, not small). Convenience ctors use
sklearn defaults (`fitIntercept=true, tol=1e-4, maxIter=1000`).

**`fitSparse(CsrMatrix x, y)`** runs the identical cd over the CSR
design's CSC mirror and agrees with the dense fit **bit for bit** —
which is exactly why it requires `fitIntercept == false` (skipping exact
zeros is bitwise-neutral; implicit-mean centering would not be). It
throws `MlException` otherwise: center densely or add an explicit
constant column. `CsrMatrix` is `cajeta.nucleo.sparse` (stdlib ≥
v0.13.0).

## LogisticRegression — the multiclass decision

- Binary targets: direct newton-cholesky fit either way.
- `K > 2`, default: **one-vs-rest** (per-class sigmoids, normalized
  probabilities).
- `K > 2`, `multinomial=true` (4-arg ctor): ONE softmax likelihood via
  damped full Newton on the symmetric K-block system. Probabilities match
  sklearn's multinomial mode (sklearn ≥ 1.7's only mode); **raw
  multinomial coefficients are gauge-fixed (centered intercepts) and NOT
  comparable to sklearn's** — compare probabilities, never coefficients.
- L2 is ON by default (`C = 1`, sklearn's muscle-memory convention) and
  never applies to intercepts. Separable data ⇒ the loud non-convergence
  warning; that's the data, not a bug.

## Hazards

- Labels must be float64-encoded integers `0..K-1`; anything else throws.
- `predict` threshold for binary is decision > 0 (≡ probability 0.5).
- Non-convergence keeps the LAST iterate and still predicts — check
  `converged()` in anything automated.
- Shrinkage trades training fit for stability: don't assert OLS-level R²
  on a regularized model (the Lasso pin's reference training R² is 0.82).
