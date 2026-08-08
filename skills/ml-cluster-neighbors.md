---
id: ml-cluster-neighbors
applies-to: [dev.cajeta.ml.cluster.KMeans, dev.cajeta.ml.neighbors.KNeighborsRegressor, dev.cajeta.ml.neighbors.KNeighborsClassifier]
title: KMeans & k-nearest neighbors — surfaces, determinism, tie-breaks, score semantics
description: The clustering/neighbors estimators — KMeans' seeded k-means++ and negative-inertia score, kNN's brute-force + zero-distance rule, and the deterministic tie-breaks this library documents where sklearn leaves them unspecified.
---

# KMeans & k-nearest neighbors

## KMeans

```cajeta
KMeans km = heap KMeans((int64) 3, (uint64) 7);   // defaults maxIter=300, tol=1e-8
km.fit(x, yIgnored);                              // y ignored (lifecycle uniformity)
Tensor<float64> centers = km.centers();           // (k, p)
Tensor<float64> labels  = km.labels();            // (n,) of the FIT
float64 wcss = km.inertia();                      // within-cluster Σ‖x−c‖²
Tensor<float64> ids = km.predict(x2);             // nearest-center id per row
```

- Init is **seeded k-means++ over the stdlib `Generator`** — deterministic
  per seed, but NOT numpy's stream: on separated data the converged
  partition/centers/inertia are init-independent (that's what the suite
  pins against sklearn); on ambiguous data different seeds may converge
  differently, same as sklearn with different `random_state`.
  There is no `n_init` — one seeded run; sweep seeds yourself if you want
  restarts.
- Empty-cluster rescue: the point farthest from its center is adopted
  (deterministic).
- **`score(x, y)` = NEGATIVE inertia of x under the fitted centers**
  (sklearn's convention) — "higher is better" means less negative. Don't
  average it with R²-style scores.
- Nearest-center ties → lowest center index.

## kNN

```cajeta
KNeighborsRegressor r = heap KNeighborsRegressor((int64) 3, /*distanceWeights=*/true);
r.fit(x, y);                                   // COPIES x and y (training set kept)
Tensor<float64> p = r.predict(q);              // (nQueries,)

KNeighborsClassifier c = heap KNeighborsClassifier((int64) 3);
c.fit(x, labels);
Tensor<float64> proba = c.predictProba(q);     // (n, K) vote fractions
```

- Brute-force euclidean (no trees/LSH) — fine for the small-n regimes this
  library targets; O(n·q·p) per predict.
- Regressor weights: `false` = uniform mean; `true` = 1/distance with
  **sklearn's zero-distance rule** — exact training-point matches take ALL
  the weight (uniformly among themselves), so predicting a training row
  returns its training target exactly.
- Classifier: uniform vote fractions; `predict` = argmax; `score` =
  accuracy.
- **Documented tie-breaks** (sklearn leaves these to implementation
  order): equal-distance neighbors → lowest training index; vote ties →
  lowest label.
- `fit` requires `n >= k`; labels are integers `0..K-1`.

## Hazards

- kNN `fit` copies the training set — memory scales with it; there is no
  index structure to reuse.
- KMeans `labels()` are the fit assignment; `predict` on the training x
  reproduces them, but only because centers are settled — don't compare
  labels across different fits without canonicalizing (cluster ids are
  arbitrary, as in sklearn).
- Feature scaling matters for both (euclidean geometry): put a scaler in
  front via `Pipeline` when features have mixed units.
