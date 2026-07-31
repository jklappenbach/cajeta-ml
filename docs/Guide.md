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

### LogisticRegression(c, tol, maxIter)
sklearn's `newton-cholesky`: IRLS Newton steps, each a `choSolve`.
Objective: `0.5‖w‖² + C·Σ log-loss` — **L2 is on by default (`C = 1`),
the sklearn convention**; pass a very large `C` for a near-unpenalized
fit. Binary targets fit directly; `K > 2` integer labels fit one-vs-rest.
`predictProba` (binary `[1−p, p]`; OvR normalized), `predict` (threshold /
argmax), `score` = accuracy. Non-convergence prints a **loud warning** and
sets `converged()` false — separable data is the classic cause.

`summary(x, y)` (binary only) gives **Wald** statistics from the penalized
Hessian with `penalized = true` — not classical inference (see
[differences](DifferencesFromSklearn.md)).

## Preprocessing & model selection

- `StandardScaler` / `MinMaxScaler` — sklearn semantics (ddof-0 scale,
  zero-variance guard), plus `inverseTransform`.
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
