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
- **Multiclass strategy is an explicit knob, not a silent default.**
  sklearn ≥ 1.7 removed `multi_class` and always fits softmax; here
  `K > 2` defaults to one-vs-rest (the v1 contract) and softmax is the
  `multinomial` constructor argument. Softmax probabilities match
  sklearn's; raw multinomial coefficients are gauge-fixed (centered
  intercepts) and are not comparable to sklearn's optimizer-path values.
- **Lasso/ElasticNet are single-target.** sklearn's `Lasso` quietly accepts
  `(n, k)` targets (fitting columns independently); the multi-task variants
  are a different model family. Here the cd models take `(n,)` and say so.
- **Dataframes cross through an explicit bridge, not duck-typing.** sklearn
  accepts a pandas DataFrame anywhere an array is expected and infers
  feature columns implicitly; here `Frames.design<R>(t, target)` makes the
  selection a visible, auditable step (float64 columns minus the target, or
  exactly the columns you name) and records `featureNames` on the `Design`.
  Nothing is silently coerced or silently dropped.
- **KMeans has one seeded run, not `n_init`.** Init is k-means++ over the
  stdlib `Generator` (deterministic per seed, not numpy's stream); sweep
  seeds yourself for restarts. `score` is negative inertia, as in sklearn.
- **kNN tie-breaks are documented.** Equal-distance neighbors break to the
  lowest training index; classifier vote ties to the lowest label — where
  sklearn's behavior is an implementation detail of its sort.
- **PCA is the full-SVD solver only** (no randomized/arpack/covariance
  paths, no whitening) — components carry sklearn's `svd_flip` signs, so
  numerics match the reference's `svd_solver="full"`.
- **Pipeline is fixed-arity and owning.** `Pipeline.of` takes up to three
  transformers plus a final estimator and CONSUMES them (`#` transfer);
  there is no ColumnTransformer/FeatureUnion and no step introspection.
- **Determinism by construction.** Splits and folds take explicit seeds;
  there is no global RNG state.
- **`crossValScore` refits the estimator instance per fold** (the protocol
  has no `clone`); its state afterward is the last fold's. sklearn clones.

## Unsupervised surface (0.6.0)

- **KMedoids is PAM (BUILD + steepest-descent SWAP), deterministic — no
  seed.** scikit-learn-extra defaults to the `alternate` method with a
  seeded heuristic init; PAM was chosen for solution quality and BUILD
  makes reproduction exact by construction. Medoid indices are reported
  sorted ascending; `score` is negative inertia (plain metric distances —
  sklearn-extra defines no `score`). Oracle: scikit-learn-extra 0.3.0
  (`method='pam', init='build'`), pinned in its own venv with
  numpy 1.26.4 / sklearn 1.5.2 — 0.3.0 predates numpy 2.
- **GaussianMixture initializes from THIS library's seeded KMeans** (or
  seeded random responsibilities) — not sklearn's k-means stream, so
  per-iteration paths differ while converged parameters match. Collapse
  is REPORTED (`collapsed()`), not just floored away.
- **AgglomerativeClustering rejects ward + metric AT CONSTRUCTION.**
  sklearn raises at fit; here ward is only constructible metric-free.
  One fit answers many `k` (`cutByCount`/`cutByHeight`), where sklearn
  refits per `n_clusters`.
- **DBSCAN matches sklearn exactly, including border order** — the
  `dbscan_inner` loop is replicated (index-order scan, LIFO expansion),
  so the order-dependent assignment of a border point reachable from two
  clusters is sklearn's, not an approximation.
- **Silhouette on 1 cluster or n clusters THROWS** (as does sklearn);
  NMI uses arithmetic normalization (sklearn's default) with no
  `average_method` knob.
- **TSNE is exact-only and `fitTransform`-only.** No Barnes-Hut (a
  recorded deferral; n > 2000 is rejected with instructions), no PCA
  init, classic learning rate 200 rather than sklearn's `'auto'` — the
  embedding distributions match in structure, not coordinates. The KL it
  reports comes from `cajeta.math.stats.Information`.
- **MDS names its variant** (`"metric"`/`"nonmetric"`) instead of a
  boolean; non-metric stress is sklearn's normalized stress-1, metric
  stress is the raw sum. No out-of-sample `transform` (sklearn 1.9 has
  none either).
- **FastICA is the parallel algorithm with logcosh/exp** (no deflation,
  no `cube`); non-convergence THROWS instead of warning.
- **NMF is NNDSVD + cyclic HALS (sklearn's `cd`) with the Frobenius
  objective only** — no `mu` solver, no beta-divergence family, no
  nndsvda/nndsvdar/random inits. Negative input is rejected NAMING the
  offending entry.
- **FactorAnalysis is EM** (sklearn uses an SVD-based ML iteration —
  same optimum, different path; compare `ΛΛᵀ` and `Ψ`, never raw
  loadings) with no rotation options.
- **LLE identifies the trivial eigenvector by VARIANCE**, not position —
  robust when the bottom eigenvalues crowd the eigensolver's floor.
  Standard LLE only (no modified/hessian/ltsa).
- **SpectralEmbedding is k-NN or RBF affinity only**, `fitTransform`-only,
  with no deterministic sign flip — the family is documented as
  deterministic up to eigenvector sign, and comparisons must use
  eigenvalues or |correlation|.

## Classification surface (0.8.0)

- **The confusion matrix is sklearn's orientation — and the literature's
  transpose.** `Metrics.confusionMatrix` puts TRUE classes on rows and
  PREDICTED on columns, exactly as sklearn does. Much of the statistics
  literature draws it the other way (predicted on rows); a reader
  comparing against a textbook should transpose first. Deliberately
  documented rather than changed.
- **LDA solvers are `svd`/`lsqr`/`eigen` with sklearn's constraint
  surface** (shrinkage only under lsqr/eigen, rejected loudly under svd,
  at CONSTRUCTION — sklearn raises at fit). Ledoit-Wolf `"auto"`
  shrinkage matches `sklearn.covariance.ledoit_wolf`.
- **The decision threshold is a `predict(x, threshold)` overload** on
  binary logistic fits — sklearn needs a separate
  `FixedThresholdClassifier` wrapper. Out-of-range thresholds and
  multiclass fits are rejected with the reason; the no-argument
  `predict` keeps the 0.5 cut.
- **L1 logistic is glmnet-style coordinate descent** (the resolved §11.2
  route) — coefficients agree with sklearn's `liblinear`/`saga` at suite
  tolerance and zeros are EXACT (asserted as equality, and monotone in
  the regularization strength).
- **`precision_recall_curve` parity is the FULL curve** (every distinct
  score a threshold, ascending, plus the terminal `(1, 0)` point with no
  threshold) as a `PrCurve` object with bounds-checked accessors —
  reading the terminal point's nonexistent threshold throws instead of
  indexing off the end of a shorter array.
- **`ClassificationReport` is a typed result, not a formatted string.**
  Per-class precision/recall/F1/support plus macro and weighted rows —
  sklearn's `output_dict=True` shape with accessors; render it yourself.
- **ForwardSelector is not `SequentialFeatureSelector`.** Same greedy
  forward direction and CV scoring, but: the inner CV is a seeded
  SHUFFLED `KFold` (sklearn defaults to unshuffled), ties break to the
  lowest feature index, and there is an unconditional early stop when no
  candidate STRICTLY improves the score — sklearn only stops early under
  its `tol` option. Selected supports agree on the pinned fixture; paths
  on degenerate data may not.
- **Nadaraya-Watson has no sklearn estimator at all** (closest relative:
  `KernelReg` in statsmodels). `KernelRegressor` is pinned against the
  closed formula by hand-computable fixtures; the far-query
  zero-total-weight case falls back to the nearest training target, a
  stated policy sklearn never had to state.
- **Encoders run over tensors, not `Table`** (the §8.6 nucleo route is
  deferred until nucleo's own encoding story settles). One-hot categories
  are sorted per column like sklearn's; the unseen-category default is
  ERROR with ignore-to-all-zeros as the explicit opt-in
  (`handle_unknown="ignore"`).
- **Search metrics are named `accuracy` / `f1` / `rocAuc` / `r2`** — a
  deliberately small, loud registry (`Scorers`), not sklearn's ~50-string
  scoring table. `rocAuc` demands an estimator that exposes
  `predictProba` and says so.

## Trees and ensembles surface (0.9.0)

- **Cross-feature ties are the one place a tree can differ from sklearn.**
  sklearn's splitter visits features in RNG order (Fisher-Yates from
  `random_state`) and strict `>` keeps the first visited; we visit in
  ascending index. Within a feature the rule is identical and deterministic
  (ascending scan, first/lowest threshold kept). Any 2-sample impure node
  ties across every separating feature, so real-data trees can diverge
  structurally while scoring equivalently — the Attrition fixture pins
  behavior (metrics at 0.03), not structure, for exactly this reason.
- **Forests are reproducible under OUR seed derivation, not sklearn's RNG
  stream** (member `i` draws rows then per-node feature subsets from
  `Generator(seed + i)`). Bootstrap and per-node draws will not match
  sklearn's for any seed; scores and structure-level properties do.
- **AdaBoost is SAMME only** — sklearn 1.9 removed SAMME.R along with the
  `algorithm` parameter. Estimator weights match sklearn exactly (the
  boosting loop is RNG-free).
- **GradientBoosting ships as the REGRESSOR only** (staged predictions
  included). sklearn's `criterion` parameter is deprecated/no-effect there;
  our stages split by plain `squared_error`. The classifier was cut per the
  plan's 6.2.4 escape hatch: `dev.cajeta.xgboost` covers classification
  boosting bit-exactly, and a teaching duplicate priced out.
- **`absolute_error` uses the weighted median** (cumulative-half walk,
  boundary ties average the middle pair) — identical to sklearn's on unit
  weights.
- **Fixture regeneration protocol (§10.2):** every reference-notebook
  number is regenerated under pinned sklearn 1.9.0 before use; on any
  disagreement the FIRST suspect is the producing library's version, not
  the tree. Reference matrices are exported C-order (`ascontiguousarray`)
  — pandas emits Fortran-order and `Npy` currently misreads it (INDEX:
  npy-fortran-order-silent-misread).

## Packaging (0.10.0)

- **The root package now mirrors sklearn's module structure**: the
  ecosystem contract (`Estimator`/`Predictor`/`Transformer`/
  `ProbClassifier`/`EstimatorFactory`), `Metrics`, `Pipeline`, and the
  `Design`/`Frames` bridge stay at `dev.cajeta.ml`; estimators live in
  `.linear`, `.discriminant`, `.tree`, `.ensemble`, `.cluster`,
  `.decompose`, `.manifold`, `.neighbors`, `.preprocess`, `.select`,
  `.inspect`. Breaking for imports, identical in behaviour; 0.9.0
  remains published for existing pins.
