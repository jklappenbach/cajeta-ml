# The cajeta-ml tour, narrated

Run it with `cajeta tour` (or `./run-tour.sh`). The tour is self-checking:
every claim below is asserted at runtime, and the process exits non-zero if
any stops being true. Source: `tour/src/dev/cajeta/ml/tour/Tour.cajeta`.

The dataset is a deterministic quadratic — `y = 2 + t − 1.5t² ± 0.01` over
60 evenly spaced `t` — so every machine sees identical numbers.

1. **Ingest.** The core API speaks `Tensor<float64>`. A dataframe column
   crosses through nucleo's zero-copy seam (`Column.asTensor()`); the tour
   round-trips the target through a `Column` to show the bytes never move.
2. **Preprocess.** `StandardScaler.fitTransform` — the tour checks the
   scaled feature means are exactly zero.
3. **Split.** `trainTestSplit` at 25% with a fixed seed: 45/15 (ceil rule),
   deterministic.
4. **LinearRegression.** Fit on the train split; held-out R² > 0.999. Then
   `summary()` prints the inference table — the recovered slope and
   curvature match the generating coefficients, the real effects are
   significant, `penalized` is false (this is classical inference).
5. **Ridge.** The regularization path α ∈ {0.001 … 100}, each judged by
   5-fold `crossValScore`. On clean data CV prefers light regularization —
   the tour checks the winner and that it still explains the data.
6. **LogisticRegression.** A label rule derived from the true curve with
   deterministic flips; newton-cholesky converges, accuracy beats the
   flips, probabilities sum to one, F1 holds up.
7. **The protocol.** Three models — `DummyRegressor`, `LinearRegression`,
   `Ridge` — driven through `Predictor`-typed variables and a generic CV
   helper: model-selection code never names a concrete type, and
   `predict()` dispatches through the interface. cajeta-xgboost's
   `XGBRegressor` adapter is the shipped cross-library proof of this seam.
8. **The Frames bridge.** A `Table<Reading>` built from columns becomes a
   `LinearRegression` fit in two lines: `Frames.design<Reading>(frame,
   "crop")` selects the float64 columns minus the target (names recorded
   on the `Design`), and the fit recovers the planted linear structure.
9. **Lasso & ElasticNet.** sklearn's exact coordinate descent: the L1
   penalty drives coefficients to EXACT zeros (`sparsityCount()`), a large
   alpha zeroes everything, `ElasticNet(l1Ratio=1)` reproduces Lasso bit
   for bit — and `fitSparse` over a `CsrMatrix` matches the dense fit bit
   for bit with the same iteration count (the `fitIntercept=false`
   contract that makes that possible is checked loudly).
10. **Multiclass.** Three bands on `t`: the explicit `multinomial` knob
    fits one softmax likelihood (probability rows sum to exactly 1); OvR
    stays the default. The accuracy check documents the honest effect of
    `C = 1` regularization on band edges.
11. **Pipelines that don't lie.** `Pipeline.of(#scaler, #ridge)` refits
    the whole chain inside each CV fold; the tour measures the difference
    against the leaky pre-scaled baseline — using Ridge on purpose,
    because affine-equivariant OLS provably cannot show the leak. Then
    `PCA(1) → OLS` is principal-component regression in two lines.
12. **Structure discovery.** Seeded KMeans (same seed → bit-identical
    centers; `score` = negative inertia, sklearn's convention) and
    distance-weighted kNN's zero-distance rule (a training row predicts
    its own target exactly).
