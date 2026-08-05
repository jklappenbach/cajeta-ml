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

---

Sections 13–17 cross into the neural half. The precision changes with them:
everything above is `float64`, everything below `float32`, and the fixture
becomes `Digits` — a deterministic 4-class, 64-feature synthetic set.

13. **Autodiff.** `GradTape` driven directly, with a case whose answer is
    checkable by hand: `f(w) = sum(relu(x @ w))` at `w = I` forwards to the
    sum of `x` (10) and backwards to `x`'s column sums (4 and 6). The tape is
    define-by-run — the ops record as they execute and `backward` replays
    them in reverse.
14. **One network, two training regimes.** The contrast worth seeing.
    `BackpropTrainer` forwards the whole net, takes ONE global backward and
    steps — the tour checks the loss halved. `SpelaTrainer` trains each layer
    against its own local objective with no backward pass spanning the
    network at all, so there is no single loss to report: the tour reads
    `lastLossAt(0)` and `lastLossAt(1)` separately, and each layer's own
    class vectors. Same data, same shape of call, fundamentally different
    regimes.
15. **Checkpoints.** A safetensors round trip: `saveStateDict` → `encode` →
    `decode`, checked **bit**-stable rather than merely close, with the
    parameter names confirmed torch-shaped (which is what lets a file torch
    wrote load into a module defined here). Safetensors is preferred because
    reading it cannot execute anything.
16. **LoRA.** Zero-init means the adapter is *exactly* a no-op before it
    learns — the tour checks the adapted forward equals the base forward.
    Then the trap: a freshly constructed adapter has a **trainable** base.
    The tour asserts that, calls `freezeBase()`, and asserts the freeze —
    because skipping that call fine-tunes the whole weight while looking
    like LoRA.
17. **Online adaptation.** `observeUnlabeled` on a strict trainer refuses a
    sample below the confidence gate and *counts* the refusal — nothing is
    pseudo-labeled. A second trainer with the gate lowered and
    `selfDistill` enabled accepts, buffers, and flushes it; flushing an
    empty buffer is a no-op rather than an error. Two trainers rather than
    one because the config transfers into the trainer, so the gate is fixed
    at construction.
18. **The clustering family.** KMedoids' centres are checked to be actual
    input rows (by index identity, not proximity); GaussianMixture's
    responsibilities sum to one and BIC picks the true component count;
    one agglomerative fit yields the full linkage matrix and answers a
    different k via `cutByCount`; DBSCAN discovers the ring count on a
    fixture no convex partition can cut — K-means converges there too,
    which is exactly why the choice of algorithm is the decision.
19. **Judging a clustering.** Silhouette scores the true labels high,
    adjusted Rand is checked to be permutation-invariant, and the
    K-means elbow curve demonstrably bends at the true k. (The suite
    additionally pins a fixture where the three internal judges
    disagree — the reason all three ship.)
20. **Embeddings & decomposition.** MDS exposes the stress it minimized;
    Isomap's `transform` is checked EXACT on a training row (a real
    out-of-sample extension — t-SNE, MDS, and SpectralEmbedding refuse to
    pretend they have one); SpectralEmbedding's kept eigenvalues are
    non-trivial; t-SNE reports its KL and the tour repeats the §9.6
    warning; NMF reaches near-zero error on an exactly-factorable
    matrix; FactorAnalysis' EM log-likelihood never decreases.
