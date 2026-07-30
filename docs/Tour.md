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
   `predict()` dispatches through the interface. This is the seam a future
   `cajeta-xgboost` adapter plugs into.
