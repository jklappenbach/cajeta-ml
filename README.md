# cajeta-ml

Classical (non-deep) machine learning for the cajeta ecosystem — the
scikit-learn/statsmodels role. Estimator objects with fit/predict lifecycles
over `Tensor<float64>` (and `Table<T>`) data:

- **The estimator protocol** — `Estimator` / `Predictor` / `Transformer`,
  the contract every ecosystem model library (cajeta-xgboost included, via
  adapter) conforms to. Owned by this library.
- **Linear models** — `LinearRegression` (QR lstsq), `Ridge` (Cholesky),
  `Lasso`/`ElasticNet` (sklearn's exact coordinate descent, sparse-input
  path), `LogisticRegression` (newton-cholesky IRLS; class/sample
  weights, `"balanced"`, a decision-threshold `predict` overload, and an
  L1 penalty by coordinate descent with EXACT zeros) — with a
  statsmodels-grade `summary()` (stderr, t, p-values, R², condition
  warning).
- **Discriminants** — `LinearDiscriminantAnalysis` (svd/lsqr/eigen,
  fixed or Ledoit-Wolf `"auto"` shrinkage, priors, `predictProba`, and
  `transform` onto the discriminant directions — a supervised reducer
  beside `PCA`) and `QuadraticDiscriminantAnalysis` (Friedman
  regularization blend).
- **Neighbors & kernels** — `KNeighborsClassifier` / `KNeighborsRegressor`
  (any `cajeta.math.distance.Metric`, symmetric distance-weighting
  options, documented tie-breaks) and `KernelRegressor` (Nadaraya-Watson;
  gaussian/epanechnikov/tricube/uniform kernels, explicit far-query
  policy).
- **Clustering** — `KMeans`, `KMedoids` (PAM, any metric),
  `GaussianMixture` (EM, four covariance types, AIC/BIC),
  `AgglomerativeClustering` (four linkages, linkage matrix, tree cutting,
  cophenetic correlation), `DBSCAN` (noise labelling, discovered count,
  k-distance elbow) — all over the `cajeta.math.distance.Metric` seam.
- **Embedding & decomposition** — `PCA`, `TSNE` (exact), `MDS` (SMACOF,
  metric/non-metric, precomputed dissimilarities), `Isomap`, `LLE`,
  `SpectralEmbedding`, `FastICA`, `NMF`, `FactorAnalysis`.
- **Cluster evaluation** — silhouette (+ per-sample), Davies-Bouldin,
  Calinski-Harabasz, adjusted Rand, NMI, K-means elbow sweep.
- **Preprocessing & encoders** — `StandardScaler`, `MinMaxScaler`,
  `OneHotEncoder` / `OrdinalEncoder` (loud unseen-category policy,
  leakage-safe inside CV), and `Pipeline` (refit-per-fold, so
  preprocessing never leaks).
- **Model selection** — seeded train/test split and k-fold (both with
  stratified variants), `RepeatedHoldout`, `GridSearch` /
  `RandomizedSearch` over the `EstimatorFactory` seam (per-combination
  failure capture, metric by name via `Scorers`), `ForwardSelector`
  (greedy CV-scored feature selection, a `Transformer`), and
  `Split.crossValScore`.
- **Metrics & reporting** — the regression/classification metric set,
  ROC-AUC, precision-recall curves with average precision (`PrCurve`),
  and the per-class `ClassificationReport` with macro/weighted rows.
- **A neural half** — define-by-run autodiff (`GradTape`), `nn` modules
  (Linear/ReLU/Sequential, attention pieces, LoRA), Adam, backprop and
  SPELA trainers, safetensors checkpoints with torch-shaped names, and
  transfer-learning freeze/import — toured in sections 13–22.

## Choosing a clustering algorithm

The suite's selection matrix (`SelectionMatrixTest`) keeps this table
honest — each method wins where it should and loses where it should:

- **Well-separated compact clusters** → any of them; `KMeans` is the
  cheapest. Blobs cannot discriminate between methods.
- **Outliers dragging centroids** → `KMedoids` (centres are data rows).
- **Unequal variances / overlapping groups** → `GaussianMixture` (the
  Bayes boundary is not the midpoint; soft responsibilities say how sure).
- **Nested or elongated non-convex shapes** → `DBSCAN` or single-linkage
  `AgglomerativeClustering`; every convex partitioner fails these.
- **Noise that should stay unassigned / unknown cluster count** →
  `DBSCAN` (noise is `-1`, the count is an output).
- **A hierarchy or many `k` from one fit** → `AgglomerativeClustering`
  (cut the tree by count or height).
- **Scale matters everywhere**: standardize mixed-unit features first —
  unscaled, the widest column decides the clustering (demonstrated in
  `DistanceScalingTest`).

Numerics ride the stdlib: `cajeta.math.linalg` (Householder QR, bidiagonal
SVD, Jacobi eigh, triangular/factor solvers), `cajeta.math.distance`, and
`cajeta.math.stats`. Tests are pinned against scikit-learn 1.9.0 /
scipy 1.18.0 / statsmodels-computed fixtures, and K-medoids against
scikit-learn-extra 0.3.0 (numpy 1.26.4 / sklearn 1.5.2 — its own venv;
0.3.0 predates numpy 2). Departures are catalogued in
`docs/DifferencesFromSklearn.md`.

## Build & test

```
cajeta build        # emits build/archive/dev.cajeta.ml-<version>.cja
./run-tests.sh      # cajeta-unit suite (resolves dev.cajeta.unit itself)
```

Spec: `specs/cajeta-ml-spec.md` in the cajeta repo. License: Apache-2.0.
