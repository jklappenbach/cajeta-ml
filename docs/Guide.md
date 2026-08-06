# cajeta-ml guide

## The estimator protocol

The contract every ecosystem model library conforms to (this library owns
it; `cajeta-xgboost` joins via an adapter):

- **`Estimator`** — `fit(x, y)` trains and stores state on the instance;
  `isFitted()`. Hyperparameters are constructor arguments — there is no
  `set_params` mutation; construct a new estimator instead.
- **`Predictor extends Estimator`** — `predict(x)`, `score(x, y)` (R² for
  regressors, accuracy for classifiers — the sklearn convention).
- **`Transformer extends Estimator`** — `transform(x)`,
  `fitTransform(x, y)`; transformers ignore `y` (lifecycle uniformity).

Model-selection code (`Split.crossValScore`) sees only `Predictor`.

Data is `Tensor<float64>`: `x (n_samples, n_features)`, `y (n,)` or
`(n, k)` for multi-output regression. Dataframe columns cross via nucleo's
zero-copy seam: `Column.fromTensor` / `column.asTensor()`.

## Models

### LinearRegression(fitIntercept)
Ordinary least squares via QR `lstsq` (sklearn's algorithm), multi-output
capable. `fitIntercept` centers X and y (the intercept is never part of the
solve). Fitted surface: `coefMatrix() (p,k)`, `interceptVector() (k,)`,
`rank()` and `singularValues()` of the solved design (numpy tolerance —
rank-deficient designs get the minimum-norm solution, and you can see it).

`summary(x, y)` → `SummaryResult`: classical inference — per-coefficient
stderr, t statistics, **exact t-distribution p-values**, R²/adjusted-R²,
F statistic, design condition number (`condWarning` above 1e8). Index 0 is
the intercept. Computed with the pseudo-inverse (statsmodels' choice), so
collinear designs warn instead of exploding.

### Ridge(alpha, fitIntercept, svdSolver)
L2-regularized least squares: `(XᵀX + αI)w = Xᵀy` by Cholesky + factor
application (sklearn's dense solver), or the SVD filter `s/(s²+α)` when
`svdSolver` — use it when the Gram matrix is near-singular. The intercept
is never regularized.

### Lasso(alpha, fitIntercept, tol, maxIter) / ElasticNet(alpha, l1Ratio, …)
L1 (and L1+L2) regularized least squares by **cyclic coordinate descent** —
sklearn's exact algorithm (residual-maintained soft-threshold updates, the
duality-gap stopping rule), so coefficients match the pinned reference and
iteration counts agree. Zeros are **exact** (`sparsityCount()`); `Lasso` IS
`ElasticNet` at `l1Ratio = 1`. The intercept is never regularized. Single
target `(n,)`. Non-convergence prints the loud warning and sets
`converged()` false. sklearn-default convenience ctors: `Lasso(alpha)` /
`ElasticNet(alpha, l1Ratio)` (fitIntercept, tol 1e-4, maxIter 1000).

### LogisticRegression(c, tol, maxIter[, multinomial])
sklearn's `newton-cholesky`: IRLS Newton steps, each a `choSolve`.
Objective: `0.5‖w‖² + C·Σ log-loss` — **L2 is on by default (`C = 1`),
the sklearn convention**; pass a very large `C` for a near-unpenalized
fit. Binary targets fit directly; `K > 2` integer labels fit one-vs-rest,
or ONE softmax likelihood with the explicit `multinomial` knob (damped
full Newton on the symmetric K-block system; probabilities match sklearn's
multinomial mode — which is sklearn ≥ 1.7's only multiclass mode).
`predictProba` (binary `[1−p, p]`; OvR normalized), `predict` (threshold /
argmax), `score` = accuracy. Non-convergence prints a **loud warning** and
sets `converged()` false — separable data is the classic cause.

`summary(x, y)` (binary only) gives **Wald** statistics from the penalized
Hessian with `penalized = true` — not classical inference (see
[differences](DifferencesFromSklearn.md)).

**Cost-aware fits (0.8.0).** Extra constructors take explicit per-class
weights (`#float64[]`), sample weights (composing multiplicatively), or
the `"balanced"` string (`n / (K·n_k)`, sklearn's rule) — each scales the
per-observation log-likelihood and IRLS weights. On a binary fit,
`predict(x, threshold)` cuts at a probability other than 0.5 (out-of-range
or multiclass use is rejected with the reason); pick the cut from the PR
curve (see Metrics & reporting). The L1 penalty is the
`(c, tol, maxIter, "l1")` constructor — glmnet-style coordinate descent
whose zeros are EXACT and monotone in strength; `"elasticnet"` adds the
mixing ratio. The plain-constructor L2 path is byte-identical to 0.7.0.

### LinearDiscriminantAnalysis / QuadraticDiscriminantAnalysis (0.8.0)
Gaussian class-conditional classifiers sharing one estimation core
(priors from frequencies or supplied explicitly and validated; pooled vs
per-class covariance; Cholesky log-densities evaluated in log space with
a max shift, so no row underflows). LDA solvers: `svd` (default), `lsqr`,
`eigen`; shrinkage — a fixed intensity or Ledoit-Wolf `"auto"` — only
under lsqr/eigen, rejected at construction under svd. QDA takes a
Friedman regularization blend `regParam ∈ [0,1]`. Both give
`predictProba` (normalized posteriors) and compose in `Pipeline` and CV;
LDA is additionally a `Transformer`: `transform` projects onto at most
`min(K−1, p)` discriminant directions — a supervised reducer beside
`PCA`. A class with fewer samples than features is reported by name (QDA
needs per-class covariances). Labels and probabilities are pinned against
sklearn 1.9.0.

### KernelRegressor(kernel, bandwidth[, metric]) (0.8.0)
Nadaraya-Watson: predictions are kernel-weighted means of training
targets, `K(d/h)` over the same metric seam as k-NN (euclidean default).
Kernels `gaussian` / `epanechnikov` / `tricube` / `uniform`; bandwidth
must be positive and is the smoothness knob, monotonically. When every
weight is zero (compact kernel or total underflow — a far query), the
prediction falls back to the NEAREST training target, a stated policy.
sklearn has no Nadaraya-Watson estimator; fixtures are hand-computable.

### KMeans(k, seed[, maxIter, tol])
Seeded k-means++ over the stdlib `Generator` (deterministic per seed — not
numpy's stream; on separated data the converged partition/centers/inertia
are init-independent and pinned against sklearn), then Lloyd.
`centers()/labels()/inertia()/predict`; `score` = −inertia (sklearn's
convention). Nearest-center ties go to the lowest index.

### KNeighborsRegressor / KNeighborsClassifier (k[, metric][, distanceWeights])
Brute-force neighbour search routed through `cajeta.math.distance.Metric`
— euclidean by default, any stdlib or caller-defined metric by
constructor, and NO distance code in this library (grep-asserted by the
suite). Weighting is symmetric across the pair since 0.8.0: uniform or
1/d on either, with sklearn's zero-distance rule (exact matches take all
the weight). Classifier: vote fractions in `predictProba`, argmax ties to
the lowest label. Neighbor ties at equal distance break to the lowest
training index (documented where sklearn's is an implementation detail).

## Preprocessing & model selection

- `StandardScaler` / `MinMaxScaler` — sklearn semantics (ddof-0 scale,
  zero-variance guard), plus `inverseTransform`.
- `PCA(nComponents)` — full-SVD solver over `LinAlg.svd`: components with
  sklearn's `svd_flip` signs, `explainedVariance`/`Ratio` (full-spectrum
  denominator), `transform`/`inverseTransform`. A `Transformer`, so
  `Pipeline.of(#pca, #ols)` is principal-component regression in one line.
- `Pipeline.of([t1[, t2[, t3]]], final)` — a `Predictor` built from up to
  three `Transformer` stages + a final estimator: `fit` chains
  `fitTransform` then fits the final stage; `predict`/`score` chain
  `transform`. Because the WHOLE chain refits inside `crossValScore`,
  per-fold preprocessing is leakage-free by construction. Stages transfer
  ownership (`#`); pipelines nest.
- `OneHotEncoder([dropFirst[, ignoreUnknown]])` / `OrdinalEncoder([orderings])`
  (0.8.0) — categorical encoding over tensors, both `Transformer`s.
  One-hot categories are SORTED per column (sklearn's order); an unseen
  category at transform is an ERROR unless `ignoreUnknown` opts into
  all-zeros. Ordinal preserves a caller-supplied category order. Fitted
  on training data only, so folds cannot leak categories.
- `Split.trainTestSplit(x, y, testFraction, seed)` — seeded permutation,
  ceil test-size rule; returns `[xTrain, xTest, yTrain, yTest]`;
  `trainTestSplitStratified` (0.8.0) preserves class proportions in both
  parts and fails LOUDLY naming any class too small to give each part a
  member.
- `KFold(k, shuffle, seed)` — sklearn fold sizes; `assignments(n)` gives a
  fold id per row. `StratifiedKFold` (0.8.0) preserves proportions per
  fold (`assignments(y)`), failing loudly on a class smaller than `k`.
- `RepeatedHoldout(n, testFraction, seed)` (0.8.0) — `n` independent
  seeded splits; results carry per-split scores plus mean and standard
  deviation, never the mean alone.
- `Split.crossValScore(est, x, y, kfold)` — per-fold `score()` of any
  `Predictor` (refit per fold; the final state is the last fold's).
- `GridSearch.run(factory, dims, x, y, foldIds, metric)` /
  `RandomizedSearch.run(…, budget, seed, …)` (0.8.0) — hyperparameter
  search over the `EstimatorFactory` seam (`create(params)` builds a
  fresh estimator per fold — no state leaks). Metrics go by name through
  `Scorers` (`accuracy`, `f1`, `rocAuc`, `r2` — unknown names throw). A
  combination that throws is RECORDED and the sweep continues; the
  `SearchResult` carries every combination, score, ok flag, `failures()`,
  and `best*` over the completed ones. A 1-D grid over `k` IS the
  K-versus-error table.
- `ForwardSelector(factory, params, nSelect, nFolds, metric, seed)`
  (0.8.0) — greedy forward feature selection by cross-validated score;
  stops at the target count or when no candidate strictly improves
  (ties to the lowest index; seeded shuffled folds, so runs reproduce).
  A `Transformer`: `transform` keeps the selected columns in ascending
  original order, `selectedCount()`/`selectedAt(i)` read the subset.

### Scaling and k-NN — a worked example

k-NN is scale-sensitive under EVERY metric: a feature measured in
thousands dominates one measured in tenths, because the distance simply
adds their contributions. Put the scaler in the same pipeline as the
estimator so cross-validation refits it per fold (no leakage):

```cajeta
Transformer sc = heap StandardScaler();
Metric man = heap Manhattan();                      // any metric composes
Predictor kn = heap KNeighborsClassifier((int64) 3, #man, true);
Pipeline pl = Pipeline.of(#sc, #kn);
pl.fit(xTrain, yTrain);
float64 acc = pl.score(xTest, yTest);
```

Without the scaler, a raw-unit feature (say, income in dollars beside age
in years) decides every neighbourhood on its own; after `StandardScaler`
each feature contributes on equal terms. This needs no k-NN API — the
pipeline IS the mechanism (§5.7).

## From the frame to the fit — `Frames` / `Design`

Any `Table<R>` (Parquet/Arrow/CSV-loaded or built in memory) becomes any
`Predictor`'s fit in two lines:

```cajeta
Design d = Frames.design<Tick>(t, "price");   // or (t, features[], target)
est.fit(d.x, d.y);                            // LinearRegression, Ridge, XGBRegressor, …
```

Selection is explicit and auditable: the default takes the
**float64-physical** columns in schema order minus the target (an
`int64`/`Instant`/`Utf8` column is never silently coerced); the explicit
overload takes the columns you name in the order you name them; and
`d.featureNames` records which frame column became which `x` column, so a
`summary()` can name its coefficients. Loud `MlException` failures: unknown
or non-float64 target/feature, zero features, a nullable selected column
(fill or drop nulls first — imputation belongs to the frame), the target
listed as a feature.

## Clustering

Five algorithms over the `cajeta.math.distance.Metric` seam, chosen by
data shape (the suite's `SelectionMatrixTest` is the executable version
of this guidance):

- `KMeans(k, seed)` — seeded k-means++, Lloyd iterations. Cheapest;
  right when clusters are compact and separated.
- `KMedoids(k[, metric])` — PAM; centres are actual data rows, robust to
  the outliers that drag a mean. Deterministic (no seed). Any metric.
- `GaussianMixture(k, seed[, covType, init, maxIter, tol, regCovar])` —
  EM, four covariance types, `predictProba` soft responsibilities,
  `aic`/`bic` for choosing k, `collapsed()` reporting, monotone
  `logLikelihoodHistory()`.
- `AgglomerativeClustering(k[, linkage[, metric]])` — ward (default,
  Euclidean-only BY CONSTRUCTION) / complete / average / single. One fit
  answers many k: `linkageMatrix()` (scipy format), `cutByCount`,
  `cutByHeight`, `leafOrder()` (dendrogram data), `copheneticCorrelation`.
- `DBSCAN(eps, minSamples[, metric])` — density clustering: noise stays
  `-1`, the cluster count is discovered, `coreMask()` distinguishes
  core/border/noise, `DBSCAN.kDistances` derives the eps-elbow.

Cluster evaluation lives in `Metrics` (below). Standardize mixed-unit
features before clustering — the widest column otherwise decides.

## Embeddings & decomposition

- `TSNE(2|3, perplexity, seed)` — exact t-SNE, `fitTransform`-only (no
  `transform` for unseen points EXISTS, so none is offered), n capped at
  2000, `klDivergence()` from the stdlib. Between-cluster distances in
  the picture are not meaningful.
- `MDS(k, seed[, "metric"|"nonmetric", nInit, maxIter, eps])` — SMACOF
  with restarts, `stress()` exposed; `fitTransformDissimilarity` embeds
  any precomputed square dissimilarity matrix.
- `Isomap(nNeighbors, k)` / `LLE(nNeighbors, k[, reg])` — manifold
  methods WITH out-of-sample `transform`; a disconnected neighbour graph
  throws, naming `nNeighbors`.
- `SpectralEmbedding(k, nNeighbors | gamma)` — Laplacian eigenmaps over
  k-NN or RBF affinity, `fitTransform`-only, trivial eigenvector dropped.
- `FastICA(k, seed[, fun, maxIter, tol])` — source separation
  (logcosh/exp); non-convergence throws.
- `NMF(k[, maxIter, tol])` — non-negative factorization (NNDSVD + HALS),
  `reconstructionError()`, negative input rejected naming the entry.
- `FactorAnalysis(k[, maxIter, tol])` — EM with per-feature noise
  variances (the distinction from PCA), monotone log-likelihood.

All spectral results are deterministic up to eigenvector sign; ICA/NMF
factors up to permutation and scale; MDS/FA up to rotation. Compare
structure (eigenvalues, distances, |correlation|, `ΛΛᵀ`), never raw
coordinates. Departures from sklearn: `docs/DifferencesFromSklearn.md`.

## Metrics

Static functions over `(n,)` tensors: `mse`/`rmse`/`mae`/`r2`;
`accuracy`, binary `precision`/`recall`/`f1`, macro variants,
`confusionMatrix`, `logLoss` (1e-15 clip), `rocAuc` (rank form with
average-rank ties — matches sklearn exactly, ties included). Unsupervised:
`silhouetteScore`/`silhouetteSamples` (throws on 1 or n clusters —
undefined must say so), `daviesBouldin`, `calinskiHarabasz`,
`adjustedRand`, `nmi`, and `kmeansElbow` for the k sweep.

`confusionMatrix` puts TRUE classes on rows, PREDICTED on columns —
sklearn's orientation, which much of the literature draws TRANSPOSED;
transpose before comparing to a textbook.

### Reporting & curves (0.8.0)

- `Metrics.precisionRecallCurve(yTrue, scores)` → `PrCurve`: one point
  per distinct score threshold (ascending) plus sklearn's terminal
  `(precision 1, recall 0)` point — `points() == thresholds() + 1`, and
  reading the terminal point's nonexistent threshold throws. This is how
  a decision threshold gets CHOSEN; `predict(x, t)` is how it gets used.
- `Metrics.averagePrecision(yTrue, scores)` — the step-function area
  `Σ (Rᵢ−Rᵢ₊₁)·Pᵢ` (sklearn's `average_precision_score`); the one-number
  summary accuracy cannot be on imbalanced problems.
- `Metrics.classificationReport(yTrue, yPred, nClasses)` →
  `ClassificationReport`: per-class precision/recall/F1/support plus
  `macro*` (classes weighted equally) and `weighted*` (by support) —
  sklearn's report shape as a typed result, matched numerically.

## Errors

Misuse throws `MlException` (recoverable): predict-before-fit, shape
mismatches, bad hyperparameters, single-class ROC.

---

# Neural networks

The torch role, in the same package. Everything above is `float64` (the
estimator protocol's precision); everything below is `float32`. The boundary is
crossed exactly once, in `Nets`, at the adapter — never smeared through layers.

Numerics are pinned against **torch 2.13.0+cpu**; deliberate departures are in
`DifferencesFromTorch.md`.

## Autodiff — `ml.grad`

`GradTape<E extends Floating>` is define-by-run reverse-mode autodiff: ops
record as they execute, `backward` replays them in reverse. `E` is `float32`
(the whole `ml.nn` stack) or `float64` (for gradient checking).

```cajeta
GradTape<float32> tape = heap GradTape<float32>();
GradTensor x = tape.leaf(xb, false);      // input: no gradient
GradTensor w = tape.leaf(weights, true);  // parameter: gradient
GradTensor loss = tape.mseLoss(tape.relu(tape.matmul(x, w)), target);
tape.backward(loss);
Tensor<float32> gw = tape.gradOf(w);
```

**Build a fresh tape every step.** The tape is one-shot and first-order by
construction, which is what makes "zero the gradients" unnecessary — there is
no stale gradient state to forget to clear. No double backward.

41 ops: elementwise arithmetic, `matmul`/`matmulBatched`, the activations
(`relu` `gelu` `tanh` `sigmoid` `softmax` `logSoftmax`), reductions (`sum`
`mean` `sumAxis` `meanAxis` `maxAxis` `rowNorm`), shape ops (`reshape`
`transpose2d` `permute` `slice` `concat`), the structured kernels (`conv2d`,
the pools, `batchNorm2d`, `layerNorm`, `embedding`, `dropout`) and the losses
(`crossEntropy` `mseLoss` `cosineSim`). Broadcast-widened ops restore input
shapes on the backward pass via `Tensor.sumTo`.

`noGradBegin()`/`noGradEnd()` bracket an inference region; `detached(node)` is
the per-node stop-gradient.

`Ops` is the **device seam** — every forward kernel routes through it, so
adding a backend does not touch `GradTape`. The CPU column is complete; the GPU
column is empty. Mixed residency is refused loudly rather than resolved with a
silent copy.

## Modules and layers — `ml.nn`

A `Module` owns `Parameter` fields, declares a forward pass over a tape, and
composes into trees. `parameters()` / `parameterNames()` walk the tree by
reflection over **own declared fields** — inherited ones are not enumerated, so
compose rather than inherit when adding parameters. Names match torch's
`state_dict` keys exactly, which is what lets a torch-written checkpoint load.

Hold submodule containers in a **named field**: that is the difference between
`blocks.0.attn.wq.weight` and an unmatchable `0.attn.wq.weight`.

Layers: `Linear`, `Conv2d`, `MaxPool2d`/`AvgPool2d`/`AdaptiveAvgPool2d`,
`BatchNorm2d`, `LayerNorm`, `MultiheadAttention`, `Embedding`, `Dropout`,
`Flatten`, `LoraLinear`, and the activations. Containers: `Sequential`.
Prebuilt in `ml.zoo`: `Mlp`, `SmallCnn`, `EncoderBlock`, `EncoderStack`.

`train()`/`eval()` propagate through the tree (Dropout samples vs is identity;
BatchNorm updates vs uses running stats). `predict(x)` is the inference
wrapper. `setTrainable(false)` freezes a parameter, honored by **both**
`Nets.gradsOf` (zeros) and every optimizer (skip) — both are needed, or
AdamW's decoupled decay still moves a "frozen" weight.

Every layer constructor takes an explicit `uint64` seed. There is no global RNG.

`NetRegressor` / `NetClassifier` wrap a module as a `Predictor`, so a network
drops into `Pipeline` and `crossValScore` like any other estimator.

## Training — `ml.train` + `ml.optim`

Two trainers, the same shape from outside.

**`BackpropTrainer`** — forward the whole network, one global backward, one
optimizer step per batch:

```cajeta
Optimizer opt = heap Adam(net.parameters(), 0.01f);
BackpropTrainer tr = heap BackpropTrainer(net, #opt, Loss.CROSS_ENTROPY, 1.0f);
TrainHistory hist = tr.fit(heap Batches(x, y, 32, 23L), 12);
```

`clipNorm <= 0` disables clipping. `stepOn(xb, yb, ps)` is public if you want
your own loop.

**`SpelaTrainer`** — each layer trains against its **own local objective**;
there is no backward pass spanning the network. Per-layer losses
(`lastLossAt(i)`), per-layer freezing (`freezeBackbone`, `setLayerTrainable`),
truncatable inference (`predictFromLayer`, `accuracyFromLayer`). Configure via
`SpelaConfig(numClasses)`; `paperExact()` selects the reference settings.

SPELA also adapts **online, without labels** — opt in with
`cfg.selfDistill = true`, then `observeUnlabeled(x)` buffers a sample if it
clears `confidenceThreshold` and `flush()` applies the buffer. Samples below
the gate are skipped, not guessed, and `skippedCount()` records them. The
config transfers into the trainer, so the gate is fixed at construction.

Optimizers: `SGD(#params, lr, momentum)`, `Adam(#params, lr)`,
`AdamW(#params, lr, weightDecay)` (decoupled decay). Schedules are pure
functions — `stepLr`, `exponentialLr`, `cosineLr`, `warmupCosineLr` — so an LR
is reproducible from `(base, step)` with no hidden counter.

`Batches(x, y, batchSize, seed)` reshuffles deterministically per epoch via
`reorder(epoch)` and preserves each sample's row shape (a CNN batch stays 4-D).

`BackpropTrainer.clipByGlobalNorm` follows torch's `clip_grad_norm_`: one
global L2 norm, one shared scale factor, returning the pre-clip norm.

## Checkpoints and LoRA — `ml.io`

```cajeta
ArrayList<String> missing = Checkpoints.load(model, "model.safetensors", strict);
Checkpoints.save(model, "out.safetensors");
```

**Prefer safetensors**, for a security reason rather than a taste one: reading
it cannot execute anything — a header length, a JSON header, raw tensor bytes.
F16/BF16 widen to f32 exactly on load; the f32 round trip is bit-stable.

`PtReader` reads torch `.pt` (a ZIP holding a pickle) through the ZIP **central
directory**, with an **allowlisted** unpickler that refuses anything outside the
tensor-rebuilding vocabulary rather than skipping it. Compressed entries are
refused. Writing `.pt` is not supported.

`Checkpoints.transposeLinearWeights(sd)` converts torch's `(out, in)` dense
weights to this library's `(in, out)`. With `strict=false`, `load` returns the
unmatched names — check the return, or a non-strict load is indistinguishable
from a successful one.

**LoRA**: `LoraLinear(in, out, rank, alpha, seed)` and the
`MultiheadAttention(embed, heads, rank, alpha, seed)` overload. `B` is
zero-initialized, so the adapter is exactly a no-op before training. Freezing is
**explicit** — call `freezeBase()` / `freezeBaseProjections()`; construction
alone leaves the base trainable. `merge()` folds `A·B` into the base weight and
refuses a second merge.

## Transformer completion — §13 (0.7.0)

- **Positional encodings** — `SinusoidalPositionalEncoding(maxLen, d)`
  (precomputed buffer, torch-elementwise, odd `d` handled) and
  `LearnedPositionalEncoding(maxLen, d, seed)` (trainable table via the
  embedding gather). Both ADD to `(B, T, E)`; `T > maxLen` throws.
- **Masks** — `Masks.causal(b, t)`, `Masks.keyPadding(lengths, tq, tk)`,
  `Masks.combine`: additive `(B, Tq, Tk)` float32, `NEG` blocks with
  exact-zero weights and exact-zero gradient paths.
- **Attention** — `MultiheadAttention.forwardMasked` /
  `forwardCross(x, memory, mask)` on one shared core;
  `lastAttentionNode()` exposes the softmaxed weights for structural
  verification.
- **Decoder** — `TransformerDecoderLayer(e, h, ff, seed).decode(...)`
  (post-norm, ReLU FFN, torch field names, dropout composed outside)
  and the `TransformerDecoder` stack. End-to-end forward and TRAINED
  loss pinned against torch.

## Transfer learning & weight import — §13.5–§13.6 (0.7.0)

- `module.setTrainable(false)` freezes a subtree; optimizers skip frozen
  parameters AND reset their state on unfreeze (fresh moments — never
  stale momentum). `Sequential.replace(i, #head)` swaps a head with the
  backbone bit-identical.
- `Checkpoints.importStateDict(model, sd, strict)` → `ImportReport`
  with BOTH reconciliation directions; `importTorch` is the strict
  default; `torchToCajeta` maps raw torch layout (fused `in_proj` split,
  Linear weights transposed); `renamePrefix` retargets keys; `.pt` files
  ride the constrained unpickler.
- **Augmentation** (`ml.data`): `AugmentPipeline(seed)` over
  `RandomHorizontalFlip`/`RandomCrop`/`RandomRotation`/`Normalize`
  (explicit stats) — stochastic transforms are inert in eval.

**Out of scope, recorded (spec §13.7):** CONTRASTIVE LEARNING is
deferred with no consumer; GRAPH NEURAL NETWORKS are not in scope —
`dev.cajeta.graph` is classical graph analysis and implies no GNN
support.
