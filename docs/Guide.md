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

### KMeans(k, seed[, maxIter, tol])
Seeded k-means++ over the stdlib `Generator` (deterministic per seed — not
numpy's stream; on separated data the converged partition/centers/inertia
are init-independent and pinned against sklearn), then Lloyd.
`centers()/labels()/inertia()/predict`; `score` = −inertia (sklearn's
convention). Nearest-center ties go to the lowest index.

### KNeighborsRegressor(k, distanceWeights) / KNeighborsClassifier(k)
Brute-force euclidean. Regressor: uniform or 1/d weights with sklearn's
zero-distance rule (exact matches take all the weight). Classifier:
vote fractions in `predictProba`, argmax ties to the lowest label.
Neighbor ties at equal distance break to the lowest training index
(documented where sklearn's is an implementation detail).

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
- `Split.trainTestSplit(x, y, testFraction, seed)` — seeded permutation,
  ceil test-size rule; returns `[xTrain, xTest, yTrain, yTest]`.
- `KFold(k, shuffle, seed)` — sklearn fold sizes; `assignments(n)` gives a
  fold id per row.
- `Split.crossValScore(est, x, y, kfold)` — per-fold `score()` of any
  `Predictor` (refit per fold; the final state is the last fold's).

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

## Metrics

Static functions over `(n,)` tensors: `mse`/`rmse`/`mae`/`r2`;
`accuracy`, binary `precision`/`recall`/`f1`, macro variants,
`confusionMatrix`, `logLoss` (1e-15 clip), `rocAuc` (rank form with
average-rank ties — matches sklearn exactly, ties included).

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
