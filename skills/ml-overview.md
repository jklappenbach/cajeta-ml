---
id: ml-overview
applies-to: [dev.cajeta.ml]
title: dev.cajeta.ml — classical ML (sklearn/statsmodels role) + neural networks (torch role) + the ecosystem estimator protocol
description: Routing map for dev.cajeta.ml — classical task → estimator table, the neural stack (grad/nn/optim/train/io), the fit/predict lifecycle every model shares, package-wide data/ownership/determinism rules, and the dead-ends.
---

# dev.cajeta.ml — orientation & routing

Two halves, one package:

- **Classical ML** (the sklearn/statsmodels role): linear models with
  statistical inference, regularized paths, clustering, neighbors, PCA,
  preprocessing, model selection, metrics. Numerics pinned against
  scikit-learn 1.9.0.
- **Neural networks** (the torch role): a runtime tensor tape, modules and
  layers, optimizers, two trainers, and torch-checkpoint interop. Numerics
  pinned against torch 2.13.0+cpu.

Plus **the estimator protocol every ecosystem model library conforms to**
(`cajeta-xgboost` joins via its `XGBRegressor` adapter). Networks join it too,
through `NetRegressor` / `NetClassifier`.

**Precision differs by half, and the boundary is explicit.** The classical
surface and the estimator protocol are `Tensor<float64>`: `x (n_samples,
n_features)` 2-D, `y (n,)` 1-D (LinearRegression/Ridge also accept `(n, k)`
multi-output), classifier labels float64-encoded integers `0..K-1`. The neural
surface is `Tensor<float32>` throughout. Conversion happens once, at the
adapter boundary in `Nets` — never smeared through the layers. Dataframes cross
via `Frames.design<R>` (see the `ml-frames-bridge` skill), never by
duck-typing.

## Which half do you want?

| You want to… | Go to | Skill |
|---|---|---|
| Fit a regression/classifier on tabular data, with inference | the table below | this skill |
| Build a network, train it, load a torch checkpoint | the neural table below | `ml-nn-modules` |
| Differentiate something yourself | `GradTape<E>` | `ml-grad` |

## Neural stack → entry point

| You want to… | Use | Skill |
|---|---|---|
| Autodiff over tensors | `GradTape<float32>` — `leaf`/`backward`/`gradOf` | `ml-grad` |
| Define a network | `Module` subclass, or `Sequential` / `ml.zoo` | `ml-nn-modules` |
| Layers | `Linear` `Conv2d` `MultiheadAttention` `BatchNorm2d` `LayerNorm` `Dropout` `Embedding` … | `ml-nn-modules` |
| Train the usual way | `BackpropTrainer(net, #opt, lossKind, clipNorm)` | `ml-training` |
| Train layerwise / adapt online without labels | `SpelaTrainer(#layers, #cfg)` — `observeUnlabeled`/`flush` | `ml-training` |
| Optimizers & LR schedules | `SGD` `Adam` `AdamW`; `Schedules.*` (pure functions) | `ml-training` |
| Minibatching | `Batches(x, y, size, seed)` — `reorder(epoch)` | `ml-training` |
| Load/save torch checkpoints | `Safetensors` (preferred) · `PtReader` · `Checkpoints` | `ml-checkpoints-lora` |
| Fine-tune cheaply | `LoraLinear`, `MultiheadAttention(..., rank, alpha, seed)` | `ml-checkpoints-lora` |
| Use a network as an estimator | `NetRegressor` / `NetClassifier` | `ml-protocol` |

## Classical ML → entry point

| You want to… | Use | Notes |
|---|---|---|
| OLS + classical inference (stderr, t, exact p-values) | `LinearRegression(fitIntercept)`, then `.summary(x, y)` | the statsmodels role |
| L2-regularized regression | `Ridge(alpha, fitIntercept, svdSolver)` | svdSolver for near-singular Grams |
| L1 / sparse coefficients | `Lasso(alpha)` or `Lasso(alpha, fitIntercept, tol, maxIter)` | exact zeros; `sparsityCount()` |
| L1+L2 blend | `ElasticNet(alpha, l1Ratio)` | `l1Ratio=1` IS Lasso, bit-for-bit |
| Sparse design matrix fit | `Lasso/ElasticNet.fitSparse(CsrMatrix, y)` | requires `fitIntercept=false` — see hazards |
| Binary / one-vs-rest classification | `LogisticRegression(c, tol, maxIter)` | newton-cholesky; L2 ON by default (`C=1`) |
| True multiclass softmax | `LogisticRegression(c, tol, maxIter, /*multinomial=*/true)` | explicit knob, never a silent switch |
| Cost-aware logistic (class/sample weights, threshold) | weight ctors, `"balanced"`, `predict(x, threshold)` | `ml-classification` |
| Sparse logistic (exact zero coefficients) | `LogisticRegression(c, tol, maxIter, "l1")` | coordinate descent — `ml-classification` |
| Gaussian discriminants (linear/curved boundary) | `LinearDiscriminantAnalysis` / `QuadraticDiscriminantAnalysis` | shrinkage, priors, LDA is a `Transformer` — `ml-classification` |
| Nonparametric regression (local means) | `KernelRegressor(kernel, bandwidth[, metric])` | Nadaraya-Watson — `ml-classification` |
| Clustering (compact clusters) | `KMeans(k, seed)` | seeded k-means++; `score` = −inertia |
| Clustering with outliers / any metric | `KMedoids(k[, metric])` | PAM; centres are data rows — `ml-clustering` |
| Soft/probabilistic clustering, model selection | `GaussianMixture(k, seed)` — `predictProba`, `aic`/`bic` | 4 covariance types; collapse reported — `ml-clustering` |
| Hierarchy, many k from one fit | `AgglomerativeClustering(k[, linkage])` — `cutByCount`/`cutByHeight` | ward/complete/average/single; scipy linkage matrix — `ml-clustering` |
| Non-convex shapes, noise, unknown k | `DBSCAN(eps, minSamples)` | noise = −1; count discovered; `kDistances` elbow — `ml-clustering` |
| Judge a clustering | `Metrics.silhouetteScore/Samples`, `daviesBouldin`, `calinskiHarabasz`, `adjustedRand`, `nmi`, `kmeansElbow` | the judges DISAGREE by design — `ml-cluster-evaluation` |
| Nearest-neighbor regression / classification | `KNeighborsRegressor` / `KNeighborsClassifier` `(k[, #metric][, distanceWeights])` | any `Metric`, symmetric weighting — `ml-classification` |
| Dimensionality reduction | `PCA(nComponents)` | a `Transformer` — drops into Pipeline |
| 2-D visualization embedding | `TSNE(2, perplexity, seed)` — `fitTransform`-only | exact O(n²), capped at 2000 — `ml-embeddings-decomposition` |
| Embed dissimilarities (incl. precomputed) | `MDS(k, seed[, "nonmetric", …])` — stress exposed | `ml-embeddings-decomposition` |
| Manifold learning | `Isomap` / `LLE` (transform unseen) · `SpectralEmbedding` (fitTransform-only) | disconnection is loud — `ml-embeddings-decomposition` |
| Source separation / parts / latent factors | `FastICA` / `NMF` / `FactorAnalysis` | indeterminacy-aware — `ml-embeddings-decomposition` |
| Feature scaling | `StandardScaler()` / `MinMaxScaler()` | sklearn semantics, `inverseTransform` |
| Train/test split, K folds, CV scores | `Split.trainTestSplit(x, y, frac, seed)` / `KFold(k, shuffle, seed)` / `Split.crossValScore(est, x, y, kf)` | deterministic per seed |
| Stratified splits/folds, repeated holdout | `Split.trainTestSplitStratified` / `StratifiedKFold` / `RepeatedHoldout` | proportions preserved or LOUD — `ml-model-selection` |
| Hyperparameter search | `GridSearch.run` / `RandomizedSearch.run` over `EstimatorFactory`, metric via `Scorers` | failure capture; 1-D grid = K-vs-error table — `ml-model-selection` |
| Categorical encoding | `OneHotEncoder` / `OrdinalEncoder` | loud unseen policy; `Transformer`s — `ml-model-selection` |
| Feature selection | `ForwardSelector(factory, params, nSelect, nFolds, metric, seed)` | greedy CV-scored; a `Transformer` — `ml-model-selection` |
| PR curve, average precision, report | `Metrics.precisionRecallCurve/averagePrecision/classificationReport` | threshold choosing + sklearn-shape report — `ml-model-selection` |
| Chain preprocessing + model, leakage-free CV | `Pipeline.of(#t1[, #t2[, #t3]], #final)` | a `Predictor`; refits whole chain per fold |
| Table<R> → design matrix | `Frames.design<R>(t, target)` | `ml-frames-bridge` skill |
| Error/score functions | `Metrics.mse/rmse/mae/r2/accuracy/precision/recall/f1(+Macro)/confusionMatrix/logLoss/rocAuc` | statics over `(n,)` tensors |
| Make YOUR model work with all of the above | implement `Predictor` | `ml-protocol` skill |

## The lifecycle (every estimator, no exceptions)

```cajeta
LinearRegression m = heap LinearRegression(true);  // hyperparams = ctor args
m.fit(x, y);                                       // state lands ON the instance
Tensor<float64> p = m.predict(x2);                 // fresh owned tensor
float64 s = m.score(x2, y2);                       // R² regressors / accuracy classifiers
```

- **Hyperparameters are constructor arguments.** There is NO `set_params`,
  NO `get_params`, NO `clone` — construct a new estimator to change them.
- `fit` returns `void` (not `self`); using `predict`/`transform`/accessors
  before `fit` throws `MlException` — always recoverable, never NaN
  propagation.
- `Split.crossValScore` **refits the same instance per fold**; its state
  afterward is the last fold's. If you need the original fit, fit a fresh
  instance after CV.
- Non-convergence (Logistic, Lasso/ElasticNet) prints a LOUD stdout
  warning, keeps the last iterate, and sets `converged()` false.

## Package-wide rules

- **Determinism by construction**: anything stochastic takes an explicit
  `uint64` seed (KFold shuffle, trainTestSplit, KMeans init). No global
  RNG. Same seed ⇒ identical results.
- **Ownership**: `fit` copies/derives what it keeps (safe to drop your x/y
  after); `predict`/`transform` return fresh owned tensors; `Pipeline.of`
  takes ownership of its stages (`#` transfer — pass fresh constructions
  or surrender with `#`).
- **Numerics are pinned**: fixtures generated by
  `tools/fixtures/gen_*.py` against scikit-learn 1.9.0 (classical) and torch
  2.13.0+cpu (neural). Deliberate deviations are listed in
  `docs/DifferencesFromSklearn.md` and `docs/DifferencesFromTorch.md`.

## Not here (dead-end avoidance)

- **No pandas-style duck-typing** — a `Table<R>` is not accepted by `fit`;
  bridge explicitly with `Frames.design<R>`.
- **No `partial_fit` / warm starts** on the classical estimators (sample
  and class weights DO exist on `LogisticRegression` since 0.8.0). Online
  adaptation exists ONLY on the neural side, as
  `SpelaTrainer.observeUnlabeled` / `flush`.
- **No GPU execution yet.** `ml.grad.Ops` is the device seam and the CPU column
  is complete; the GPU column is empty. The seam exists so adding a backend
  does not touch `GradTape` — but today everything runs on CPU.
- **No higher-order gradients** — the tape is first-order by construction: no
  double backward, no `create_graph`.
- **No RNN / LSTM / GRU layers**, and no distributed or multi-GPU training.
- **No Lasso/ElasticNet multi-target** — `(n,)` only (sklearn's multitask
  variants are a different model family, not implemented).
- **No decision trees / forests / gradient boosting here** — boosted trees
  live in `dev.cajeta.xgboost` (its `XGBRegressor` conforms to this
  package's `Predictor`).
- **No ColumnTransformer / FeatureUnion**; `Pipeline.of` takes at most 3
  transformers + a final estimator.
- Statistical special functions (`betainc`, `tCdf`, `erf`, …) are
  `cajeta.math.stats.Stats`, not this package.

## Hazards (the expensive ones)

- `fitSparse` refuses `fitIntercept=true` loudly — zero-skipping is
  bitwise-neutral, implicit centering is not; center densely or add a
  constant column.
- `summary()` on LogisticRegression is **Wald statistics from a penalized
  Hessian** (`penalized=true`) — not classical p-values. OLS `summary()`
  IS classical.
- KMeans `score` is **negative inertia** (sklearn's convention) — bigger
  is better only in the "less negative" sense.
