---
id: ml-selection-pipelines
applies-to: [dev.cajeta.ml.Split, dev.cajeta.ml.KFold, dev.cajeta.ml.Pipeline, dev.cajeta.ml.StandardScaler, dev.cajeta.ml.MinMaxScaler, dev.cajeta.ml.PCA, dev.cajeta.ml.Metrics]
title: Model selection & pipelines — splits, K-fold CV, Pipeline, scalers, PCA, metrics
description: The deterministic split/CV machinery, Pipeline's leakage-free-by-construction contract (and the affine-equivariance trap when testing it), scaler/PCA transformer surfaces, and the Metrics reference.
---

# Model selection & pipelines

## Splits and CV — deterministic, seeded

```cajeta
Tensor<float64>[] parts = Split.trainTestSplit(x, y, 0.25, (uint64) 42);
// parts = [xTrain, xTest, yTrain, yTest]; ceil test-size rule
KFold kf = heap KFold((int64) 5, /*shuffle=*/true, (uint64) 7);
Tensor<float64> scores = Split.crossValScore(est, x, y, kf);  // (k,) of score()
```

- `KFold(k, shuffle, seed)`: sklearn fold sizes (first `n mod k` folds get
  one extra row); `assignments(n)` gives a `(n,)` fold-id tensor if you
  need masks yourself. `k >= 2`.
- `crossValScore` accepts ANY `Predictor` — including `Pipeline` and
  cross-library conformers — and **refits the instance per fold** (no
  `clone` in the protocol; final state = last fold's).

## Pipeline — leakage-free CV by construction

```cajeta
StandardScaler sc = heap StandardScaler();
Ridge rr = heap Ridge(1.0, true, false);
Pipeline pl = Pipeline.of(#sc, #rr);          // takes OWNERSHIP (#)
Tensor<float64> honest = Split.crossValScore(pl, x, y, kf);
```

`Pipeline.of(...)` arities: `(final)`, `(t1, final)`, `(t1, t2, final)`,
`(t1, t2, t3, final)` — transformers then one final `Predictor`. `fit`
chains `fitTransform` left→right then fits the final stage;
`predict`/`score` chain `transform`. Because the WHOLE chain refits
inside each CV fold, per-fold preprocessing cannot leak test-fold
statistics. Pipelines nest (a Pipeline is a legal final stage).
`transformChain(x)` runs just the fitted transformers.

**The affine-equivariance trap** (a real test failure): plain OLS is
affine-equivariant, so per-fold vs global scaling provably CANNOT change
its predictions — a leakage demo using OLS will show nothing. Use a
model whose geometry depends on scaling (Ridge, Lasso, kNN, KMeans) to
observe the difference.

## Transformers

- `StandardScaler` — per-column `(x − μ)/σ`, population (ddof-0) σ,
  zero-variance columns scale by 1 (constants → 0). `inverseTransform`.
- `MinMaxScaler` — sklearn semantics, `inverseTransform`.
- `PCA(nComponents)` — center + full SVD (`LinAlg.svd`); components carry
  sklearn's `svd_flip` sign convention (largest-|loading| positive);
  `explainedVariance()` = `s²/(n−1)`, `explainedVarianceRatio()` against
  the FULL spectrum; `transform`/`inverseTransform` (lossy for k < p).
  Being a `Transformer`, `Pipeline.of(#pca, #ols)` is principal-component
  regression in one line. No whitening; full solver only (no randomized/
  arpack).

## Metrics — statics over `(n,)` float64 tensors

`mse`, `rmse`, `mae`, `r2`; `accuracy`; binary `precision`/`recall`/`f1`
+ `precisionMacro`/`recallMacro`/`f1Macro`; `confusionMatrix` (returns
`Tensor<int64>`); `logLoss(yTrue, pPos)` (1e-15 clip); `rocAuc(yTrue,
scores)` — the rank form with average-rank ties, matching sklearn exactly
(single-class input throws).

## Hazards

- `Pipeline.of` consumes its stages (`#` formals) — construct inline or
  surrender locals with `#`; you cannot reuse a stage object after.
- Never pre-scale the full matrix and then CV the bare model when
  measuring generalization — that's the leak Pipeline exists to prevent.
- KMeans/kNN also conform to `Predictor`, so they ride `crossValScore` —
  but read their `score` semantics first (`ml-cluster-neighbors`).
