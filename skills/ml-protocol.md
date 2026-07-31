---
id: ml-protocol
applies-to: [dev.cajeta.ml.Estimator, dev.cajeta.ml.Predictor, dev.cajeta.ml.Transformer]
title: The estimator protocol — Estimator / Predictor / Transformer, and how to conform
description: The three interfaces every ecosystem model library implements, the exact contract each method carries, and a worked conformer (cajeta-xgboost's XGBRegressor) — including the # ownership shapes that make interface conformance compile.
---

# The estimator protocol

Three interfaces, owned by `dev.cajeta.ml`, that make model-selection code
generic: anything conforming drops into `Split.crossValScore`, `Pipeline`,
and any utility typed against them. `cajeta-xgboost`'s `XGBRegressor` is
the shipped cross-library proof.

```cajeta
public interface Estimator {
    void fit(Tensor<float64> x, Tensor<float64> y);
    boolean isFitted();
}
public interface Predictor extends Estimator {
    #Tensor<float64> predict(Tensor<float64> x);
    float64 score(Tensor<float64> x, Tensor<float64> y);
}
public interface Transformer extends Estimator {
    #Tensor<float64> transform(Tensor<float64> x);
    #Tensor<float64> fitTransform(Tensor<float64> x, Tensor<float64> y);
}
```

## The contract, method by method

- `fit(x, y)` — `x (n, p)` 2-D; trains and stores ALL fitted state on the
  instance. Transformers accept `y` and ignore it (sklearn's
  `fit(X, y=None)` convention — one lifecycle for models and transformers
  is what makes `Pipeline` possible). Validate shapes FIRST and throw
  `MlException` before mutating any state, so a failed fit leaves
  `isFitted()` false.
- `isFitted()` — false until a `fit` COMPLETES successfully.
- `predict(x)` — `(n,)` float64 (or `(n, k)` for multi-output fits); a
  FRESH owned tensor (`#Tensor` return). Throws `MlException` before fit
  or on a p mismatch with the fitted design.
- `score(x, y)` — the sklearn convention: **R² for regressors, accuracy
  for classifiers** (use `Metrics.r2` / `Metrics.accuracy` — don't
  hand-roll). `crossValScore` aggregates exactly this.
- `transform` / `fitTransform` — `fitTransform(x, y)` MUST equal
  `fit(x, y)` then `transform(x)`.

Classifier extras (`predictProba`) are concrete-class surface, not part of
the protocol — model-selection code cannot assume them.

## Writing a conformer — the worked example

`dev.cajeta.xgboost.api.XGBRegressor` wraps a train/predict pipeline that
speaks arrays, not tensors. The adapter pattern:

```cajeta
public final class XGBRegressor implements Predictor {
    private Params params;      // snapshot at construction
    private Model model;        // fitted state
    private int64 nFeatures;
    private boolean fitted;

    public XGBRegressor(Params p) {
        Params own = heap Params();
        own.rounds = p.rounds;          // copy field-by-field: the ctor
        /* … */                          // arg is a borrow, don't store it
        this.params #= own;              // owned field stores use #=
        this.fitted = false;
        return;
    }

    public void fit(Tensor<float64> x, Tensor<float64> y) {
        if (x.ndim() != 2) { throw heap MlException("…: x must be 2-D (n, p)"); }
        int64 n = x.shapeAt(0);
        if (y.ndim() != 1 || y.shapeAt(0) != n) { throw heap MlException("…"); }
        // adapt tensors → the native input, run the engine, store the model
        Model m = GBDT.fit(x, label, n, x.shapeAt(1), this.params);
        this.model #= m;
        this.fitted = true;
        return;
    }

    public boolean isFitted() { return this.fitted; }

    public #Tensor<float64> predict(Tensor<float64> x) {
        if (!this.fitted) { throw heap MlException("…: fit first"); }
        // widen/adapt engine output into a fresh (n,) float64 tensor
        return Tensor.of<float64>(wide, sh);
    }

    public float64 score(Tensor<float64> x, Tensor<float64> y) {
        Tensor<float64> p = this.predict(x);
        return Metrics.r2(y, p);        // regressor convention
    }
}
```

Reference conformers to copy from: `DummyRegressor` (minimal Predictor,
the protocol's executable spec), `IdentityTransformer` (minimal
Transformer), `StandardScaler` (real Transformer with inverse).

## Conformance gotchas (each one cost a real debugging session)

- **`#Tensor<float64>` return through an interface** needs toolchain ≥
  v0.12.0 (the iface-generic-returns ABI fix). On older toolchains,
  dispatch through a `Predictor`-typed variable corrupts arguments.
- **Owned fields**: store with `#=`; plain `=` into an owned field of a
  ctor-arg borrow dangles. Copy ctor args you intend to keep.
- **Don't cache borrowed inputs.** `fit`'s x/y are borrows; copy
  (`x.copy()`) anything kept past the call (kNN does exactly this).
- **`crossValScore` refits your instance per fold** — `fit` must fully
  RESET state, not accumulate (replace the model wholesale; recompute
  nClasses; etc.).
- Expose the native surface alongside — the adapter should offer an
  escape hatch (`XGBRegressor.model()`) so protocol users can still reach
  engine-specific features (serialization, importances).
