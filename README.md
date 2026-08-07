# cajeta-ml

Machine learning for the cajeta ecosystem, in two halves:

- **Classical** — the scikit-learn / statsmodels role over
  `Tensor<float64>`: estimators with fit/predict lifecycles, linear models
  with real inference, discriminants, neighbors, clustering, embeddings,
  preprocessing, model selection, and metrics.
- **Neural** — the torch role over `Tensor<float32>`: define-by-run
  autodiff, modules and layers through transformers, optimizers, two
  trainers, and torch-checkpoint interop.

The precision boundary is crossed exactly once, at the estimator adapter.
This library also owns the ecosystem **estimator protocol** that sibling
model libraries (cajeta-xgboost, via adapter) conform to.

## Documentation

| Page | What it covers |
|---|---|
| [Documentation index](docs/README.md) | Orientation for both halves |
| [Guide](docs/Guide.md) | The protocol, every model and utility, API shapes |
| [Tour](docs/Tour.md) | The runnable, self-checking walkthrough — 28 sections, `cajeta tour` |
| [Differences from scikit-learn](docs/DifferencesFromSklearn.md) | Where and why the classical half deliberately diverges |
| [Differences from PyTorch](docs/DifferencesFromTorch.md) | The same for the neural half, plus what is honestly not implemented |

Agent-facing **skills** ship inside the `.cja` (`skills/*.md`, indexed):
`cajeta search-skill dev.cajeta.ml` / `list-skills` / `get-skills` (or the
same over cajeta-mcp) route a coding agent to the right estimator, the
protocol contract, and the hazards — start at `ml-overview`.

## Quick start

```cajeta
LinearRegression lr = heap LinearRegression(true);
lr.fit(xTrain, yTrain);                        // Tensor<float64> (n,p), (n,)
float64 r2 = lr.score(xTest, yTest);
SummaryResult s = lr.summary(xTrain, yTrain);  // stderr, t, p-values, R², F
```

Dataframe columns cross via nucleo's zero-copy seam (`Column.asTensor()` /
`Column.fromTensor`), and `Frames.design<T>(frame, target)` turns a
`Table<T>` into a named design matrix in one line.

## The estimator protocol

- **`Estimator`** — `fit(x, y)` trains and stores state on the instance;
  `isFitted()`. Hyperparameters are constructor arguments — no `set_params`
  mutation; construct a new estimator instead.
- **`Predictor extends Estimator`** — `predict(x)`, `score(x, y)` (R² for
  regressors, accuracy for classifiers).
- **`Transformer extends Estimator`** — `transform(x)`, `fitTransform(x, y)`.

Model-selection code (`Split.crossValScore`, the searches) sees only
`Predictor` — it never names a concrete type.

## The classical half

- **Linear models** — `LinearRegression` (QR lstsq), `Ridge` (Cholesky or
  SVD filter), `Lasso`/`ElasticNet` (sklearn's exact coordinate descent,
  EXACT zeros, sparse-input path), `LogisticRegression` (newton-cholesky
  IRLS; multinomial knob, class/sample weights, `"balanced"`, a
  decision-threshold `predict` overload, L1/elastic-net by coordinate
  descent) — with a statsmodels-grade `summary()` (stderr, t, p-values,
  R², condition warning).
- **Discriminants** — `LinearDiscriminantAnalysis` (svd/lsqr/eigen, fixed
  or Ledoit-Wolf `"auto"` shrinkage, priors, `predictProba`, and
  `transform` onto the discriminant directions — a supervised reducer
  beside `PCA`) and `QuadraticDiscriminantAnalysis` (Friedman
  regularization blend).
- **Neighbors & kernels** — `KNeighborsClassifier` / `KNeighborsRegressor`
  (any `cajeta.math.distance.Metric`, distance weighting, documented
  tie-breaks) and `KernelRegressor` (Nadaraya-Watson; four kernels,
  explicit far-query policy).
- **Clustering** — `KMeans`, `KMedoids` (PAM, any metric),
  `GaussianMixture` (EM, four covariance types, AIC/BIC),
  `AgglomerativeClustering` (four linkages, tree cutting, cophenetic
  correlation), `DBSCAN` (noise labelling, k-distance elbow) — all over
  the `Metric` seam. Evaluation: silhouette (+ per-sample),
  Davies-Bouldin, Calinski-Harabasz, adjusted Rand, NMI, elbow sweep.
- **Embedding & decomposition** — `PCA`, `TSNE` (exact), `MDS` (SMACOF,
  metric/non-metric), `Isomap`, `LLE`, `SpectralEmbedding`, `FastICA`,
  `NMF`, `FactorAnalysis`.
- **Preprocessing & encoders** — `StandardScaler`, `MinMaxScaler`,
  `OneHotEncoder` / `OrdinalEncoder` (loud unseen-category policy), and
  `Pipeline` (refit-per-fold, so preprocessing never leaks).
- **Model selection** — seeded train/test split and k-fold (both with
  stratified variants), `RepeatedHoldout`, `GridSearch` /
  `RandomizedSearch` over the `EstimatorFactory` seam, `ForwardSelector`
  (greedy CV-scored feature selection), and `Split.crossValScore`.
- **Metrics & reporting** — the regression/classification metric set,
  ROC-AUC, precision-recall curves with average precision (`PrCurve`),
  and the per-class `ClassificationReport` with macro/weighted rows.

Picking a clustering algorithm: compact blobs → `KMeans`; outliers
dragging centroids → `KMedoids`; overlapping unequal-variance groups →
`GaussianMixture`; non-convex shapes → `DBSCAN` or single-linkage
agglomerative; unknown cluster count or noise that should stay unassigned
→ `DBSCAN`; a whole hierarchy from one fit → `AgglomerativeClustering`.
Standardize mixed-unit features first — unscaled, the widest column
decides the clustering. The suite's selection matrix keeps this table
honest; the [Guide](docs/Guide.md) has the full version.

## The neural half

- **Autodiff** — `GradTape` define-by-run tape, per-step and one-shot, so
  there is no `zero_grad` to forget.
- **Modules** — `Linear`, `Conv2d`, `BatchNorm2d`, pooling, `Dropout`,
  `LayerNorm`, `Embedding`, sinusoidal/learned positional encodings,
  `MultiheadAttention` (self- and cross-attention, masking),
  `TransformerDecoder`, `LoraLinear`, and `Sequential` — plus a small
  zoo (`Mlp`, `SmallCnn`, `EncoderStack`). Every source of randomness
  takes an explicit seed: same seed, bit-identical run, any machine.
- **Training** — `SGD` / `Adam` / `AdamW` (pure-function LR schedules),
  `BackpropTrainer` and the online `SpelaTrainer` (confidence-gated
  self-distillation that counts what it skips), and a seeded, eval-inert
  data-augmentation pipeline (crop/flip/rotate/normalize).
- **Checkpoints & transfer** — safetensors read/write with torch-shaped
  names, a `.pt` unpickler, `Checkpoints.transposeLinearWeights` for
  torch-authored weights, pretrained-weight import with a full
  `ImportReport` reconciliation, and explicit freeze/unfreeze honored by
  both gradients and optimizers.

`NetClassifier` / `NetRegressor` adapt a network into the estimator
protocol — the one place `float32` meets `float64`.

## Numerics & pins

Numerics ride the stdlib: `cajeta.math.linalg` (Householder QR, bidiagonal
SVD, Jacobi eigh, triangular/factor solvers), `cajeta.math.distance`, and
`cajeta.math.stats`. The classical half is pinned against scikit-learn
1.9.0 / scipy 1.18.0 fixtures (K-medoids against scikit-learn-extra
0.3.0 in its own venv); the neural half against torch 2.13.0+cpu —
optimizer trajectories and training curves step by step, not merely at
convergence. The inference summaries (`summary()`) use
statsmodels-*equivalent* closed formulas verified against those same
scipy pins — no fixture in this repo is statsmodels-computed; the
ecosystem's statsmodels oracle is pinned at **0.14.6** and lives in
`dev.cajeta.timeseries` (cajeta-timeseries), whose domain statsmodels
actually owns. Deliberate departures are catalogued in the two
[differences](docs/DifferencesFromSklearn.md)
[pages](docs/DifferencesFromTorch.md).

## Build, test, tour

```
cajeta build        # emits build/archive/dev.cajeta.ml-<version>.cja
./run-tests.sh      # cajeta-unit suite (resolves dev.cajeta.unit itself)
./run-tour.sh       # the self-checking 28-section tour (also: cajeta tour)
```

Spec: `specs/cajeta-ml-spec.md` in the cajeta repo. License: Apache-2.0.
