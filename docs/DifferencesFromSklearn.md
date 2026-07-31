# Differences from scikit-learn

Recognizable, not faithful: where a scikit-learn convention encodes a
genuine mistake — or just Python — this library corrects it. The
numerics themselves are pinned against sklearn 1.9.0 in the test suite.

- **No `set_params` / `get_params` / `clone`.** Hyperparameters are
  constructor arguments; build a new estimator to change them. (Cajeta has
  constructors and no `**kwargs` reflection to work around.)
- **Coefficients are `(p, k)` tensors** (`coefMatrix()`), one column per
  output — not sklearn's transposed `(k, p)` `coef_` array attribute.
- **`fit` returns nothing** (sklearn returns `self` for chaining). The
  lifecycle is explicit: `fit`, then `predict`/`transform`.
- **Non-convergence is LOUD.** sklearn's `ConvergenceWarning` is easy to
  silence and easier to miss; here it prints unconditionally and
  `converged()` reports it. The last iterate is still returned.
- **Logistic regularization is documented at the door.** Same default as
  sklearn (`C = 1`, L2 on) for muscle memory — but `summary()` on a
  penalized fit sets `penalized = true` and its Wald statistics are labeled
  what they are, instead of letting a statistician mistake them for
  classical p-values.
- **Inference is first-class.** `LinearRegression.summary()` gives the
  statsmodels table (stderr, t, exact t-distribution p-values via the
  stdlib's `betainc`, R²/adjusted, F, condition number) — no second
  library needed.
- **Errors throw typed exceptions** (`MlException`), not warnings +
  NaN propagation: predict-before-fit, shape mismatches, single-class AUC.
- **Dataframes cross through an explicit bridge, not duck-typing.** sklearn
  accepts a pandas DataFrame anywhere an array is expected and infers
  feature columns implicitly; here `Frames.design<R>(t, target)` makes the
  selection a visible, auditable step (float64 columns minus the target, or
  exactly the columns you name) and records `featureNames` on the `Design`.
  Nothing is silently coerced or silently dropped.
- **Determinism by construction.** Splits and folds take explicit seeds;
  there is no global RNG state.
- **`crossValScore` refits the estimator instance per fold** (the protocol
  has no `clone`); its state afterward is the last fold's. sklearn clones.
