# cajeta-ml

Classical (non-deep) machine learning for the cajeta ecosystem — the
scikit-learn/statsmodels role. Estimator objects with fit/predict lifecycles
over `Tensor<float64>` (and `Table<T>`) data:

- **The estimator protocol** — `Estimator` / `Predictor` / `Transformer`,
  the contract every ecosystem model library (cajeta-xgboost included, via
  adapter) conforms to. Owned by this library.
- **Linear models** — `LinearRegression` (QR lstsq), `Ridge` (Cholesky),
  `LogisticRegression` (newton-cholesky IRLS) — with a statsmodels-grade
  `summary()` (stderr, t, p-values, R², condition warning).
- **Preprocessing, model selection, metrics** — scalers, train/test split,
  k-fold CV, and the regression/classification metric set.

Numerics ride the stdlib: `cajeta.math.linalg` (Householder QR, bidiagonal
SVD, triangular/factor solvers) and `cajeta.math.stats`. Tests are pinned
against scikit-learn 1.9.0 / scipy 1.18.0 / statsmodels-computed fixtures.

## Build & test

```
cajeta build        # emits build/archive/dev.cajeta.ml-<version>.cja
./run-tests.sh      # cajeta-unit suite (resolves dev.cajeta.unit itself)
```

Spec: `specs/cajeta-ml-spec.md` in the cajeta repo. License: Apache-2.0.
