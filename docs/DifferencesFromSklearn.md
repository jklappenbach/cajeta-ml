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
