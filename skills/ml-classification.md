---
id: ml-classification
applies-to: [dev.cajeta.ml.discriminant.LinearDiscriminantAnalysis, dev.cajeta.ml.discriminant.QuadraticDiscriminantAnalysis, dev.cajeta.ml.neighbors.KNeighborsClassifier, dev.cajeta.ml.linear.LogisticRegression, dev.cajeta.ml.neighbors.KernelRegressor]
title: Classification — discriminants (LDA/QDA), k-NN with metrics & weights, cost-aware & L1 logistic, kernel regression
description: The 0.8.0 classification surface — Gaussian discriminants with shrinkage and LDA-as-Transformer, the metric-agnostic k-NN classifier with symmetric distance weighting, class/sample weights + decision thresholds + L1 coordinate descent on LogisticRegression, and Nadaraya-Watson kernel regression with its stated far-query policy.
---

# Classification (0.8.0 surface)

## Discriminants — LDA and QDA

```cajeta
LinearDiscriminantAnalysis lda = heap LinearDiscriminantAnalysis();  // svd
lda.fit(x, y);                       // y: float64-encoded labels 0..K-1
Tensor<float64> post = lda.predictProba(xTest);   // normalized posteriors
Tensor<float64> proj = lda.transform(x);          // (n, <= min(K-1, p))
QuadraticDiscriminantAnalysis qda = heap QuadraticDiscriminantAnalysis(0.1);
```

- Ctors: `()` (svd) · `(solver)` — `"svd"`/`"lsqr"`/`"eigen"` ·
  `(solver, shrinkage)` fixed intensity `[0,1]` · `(solver, "auto")`
  Ledoit-Wolf from the data · `(#priors)` explicit class priors
  (validated at fit: must sum to 1, no negatives). Shrinkage under
  `"svd"` is rejected AT CONSTRUCTION.
- QDA: `()` · `(regParam)` Friedman blend `[0,1]` · `(regParam,
  storeCovariance)`. A class with fewer samples than features is
  reported BY NAME (per-class covariance needs them).
- Log-densities are evaluated in log space with a max shift — no row
  underflows, ever. Posteriors are pinned against sklearn 1.9.0.
- **LDA is a `Transformer`**: supervised dimensionality reduction beside
  `PCA` — `Pipeline.of(#lda, #knn)` is legal and leakage-free under CV.
- LDA's boundary is linear (pooled covariance), QDA's is curved
  (per-class): when classes share shape use LDA, when spreads differ use
  QDA — the suite pins a fixture where the boundary curvature decides.

## k-NN classification — one metric seam, symmetric weighting

```cajeta
KNeighborsClassifier kn = heap KNeighborsClassifier((int64) 5);          // euclidean
KNeighborsClassifier km = heap KNeighborsClassifier((int64) 5, heap Manhattan());
KNeighborsClassifier kw = heap KNeighborsClassifier((int64) 5, /*distanceWeights=*/true);
KNeighborsClassifier kb = heap KNeighborsClassifier((int64) 5, heap Chebyshev(), true);
```

- Any `cajeta.math.distance.Metric` (stdlib or your own) — the library
  itself contains NO distance code (grep-asserted by the suite).
- `distanceWeights` mirrors the regressor exactly (1/d votes, sklearn's
  zero-distance rule: exact matches take all the weight).
- Ties: vote ties break to the LOWEST label; equal-distance neighbor
  ties to the lowest training index — documented, deterministic.
- k-NN is scale-sensitive under EVERY metric — put `StandardScaler` in
  the same `Pipeline` (worked example in `docs/Guide.md`).

## Cost-aware & sparse logistic

```cajeta
// class weights (explicit / balanced), sample weights:
LogisticRegression w = heap LogisticRegression(1.0, 0.00000001,
    (int64) 500, false, #weights);          // #float64[] per class
LogisticRegression b = heap LogisticRegression(1.0, 0.00000001,
    (int64) 500, false, "balanced");        // n / (K * n_k)
// decision threshold (binary fits only; (0,1) enforced):
Tensor<float64> hard = m.predict(xTest, 0.35);
// L1 by coordinate descent — zeros are EXACT:
LogisticRegression l1 = heap LogisticRegression(0.5, 0.00000001,
    (int64) 2000, "l1");
```

- Class and sample weights scale the log-likelihood and IRLS weights;
  they compose multiplicatively. `"balanced"` is sklearn's rule.
- `predict(x, threshold)` rejects thresholds outside `(0,1)` and
  multiclass fits — with the reason. Choose the cut from
  `Metrics.precisionRecallCurve` (see `ml-model-selection`).
- L1 is glmnet-style coordinate descent: zeros are exact (assert with
  `== 0.0`, not a tolerance) and monotone in strength; agrees with
  sklearn's `liblinear`/`saga` at suite tolerance. `"elasticnet"` ctor
  adds the mixing ratio. The plain-ctor L2 path is untouched from 0.7.0.

## Kernel regression (Nadaraya-Watson)

```cajeta
KernelRegressor kr = heap KernelRegressor("gaussian", 0.5);   // or a #Metric ctor
```

- Kernels: `gaussian` / `epanechnikov` / `tricube` / `uniform` (the
  compact three vanish past `u = 1`); bandwidth > 0 is THE smoothness
  knob, monotonically.
- Far-query policy is STATED: when every kernel weight is zero (compact
  kernel, or gaussian underflow), the prediction falls back to the
  nearest training target — never a silent NaN.
- sklearn has no Nadaraya-Watson estimator; fixtures are hand-computed.

## Hazards

- Discriminant `predictProba` needs the fit's class order — labels are
  float64-encoded `0..K-1`, and priors you pass are in THAT order.
- `"balanced"` weights come from the TRAINING split's frequencies inside
  CV — fit inside the fold (Pipeline does this for free).
- A curve-chosen threshold uses `>=` semantics; `predict(x, t)` cuts at
  `> t`. Apply a hair below the chosen threshold to reproduce the
  curve's cut exactly (the tour does).
