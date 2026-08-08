---
id: ml-clustering
applies-to: [dev.cajeta.ml.cluster.KMedoids, dev.cajeta.ml.cluster.GaussianMixture, dev.cajeta.ml.cluster.AgglomerativeClustering, dev.cajeta.ml.cluster.DBSCAN]
title: The clustering family — KMedoids, GaussianMixture, AgglomerativeClustering, DBSCAN, and how to choose
description: The 0.6.0 clustering surface beyond KMeans — medoid robustness, EM mixtures with four covariance types and collapse reporting, hierarchical fits that answer many k, density clustering that labels noise — plus the selection table and the ownership/determinism rules they share.
---

# The clustering family

`KMeans` is covered in `ml-cluster-neighbors`. This skill is the rest, and
the choice between them — which is the actual decision.

## Choosing (the suite's SelectionMatrixTest keeps this honest)

| Data looks like… | Use | Why |
|---|---|---|
| Compact, separated | `KMeans(k, seed)` | cheapest; everything wins here |
| Outliers present | `KMedoids(k)` | centres are DATA ROWS — means get dragged, medoids don't |
| Overlapping / unequal variances | `GaussianMixture(k, seed)` | the Bayes boundary is not the midpoint; soft membership |
| Nested / elongated non-convex | `DBSCAN(eps, minSamples)` or `AgglomerativeClustering(k, "single")` | convex partitions cannot cut rings or moons |
| Noise should stay unassigned | `DBSCAN` | noise = `-1`, count DISCOVERED |
| Want a hierarchy / many k cheaply | `AgglomerativeClustering` | one fit, `cutByCount`/`cutByHeight` |

Scale first (`StandardScaler`) when features mix units — unscaled, the
widest column IS the clustering.

## KMedoids

```cajeta
KMedoids km = heap KMedoids((int64) 3);                    // Euclidean
KMedoids kc = heap KMedoids((int64) 3, heap Chebyshev());  // any Metric — OWNERSHIP TRANSFERS
km.fit(x, yIgnored);
int64[] rows = km.medoidIndices();     // ascending; medoids() are copies of those rows
```

- PAM (greedy BUILD + steepest-descent SWAP) — **deterministic, no seed**;
  ties break to lowest index. `inertia()` is the sum of PLAIN metric
  distances (not squared); `score` its negative.
- The metric constructor **takes ownership** (`#Metric`) — pass a fresh
  `heap Manhattan()` or hand over a local you will not reuse.

## GaussianMixture

```cajeta
GaussianMixture gm = heap GaussianMixture((int64) 3, (uint64) 7);  // full cov, kmeans init
GaussianMixture gd = heap GaussianMixture((int64) 3, (uint64) 7,
    "diag", "random", (int64) 100, 0.001, 0.000001);  // covType, init, maxIter, tol, regCovar
gm.fit(x, yIgnored);
Tensor<float64> resp = gm.predictProba(x);   // (n, k), rows sum to 1 — the point
float64 quality = gm.bic(x);                 // lower better; sweep k with aic/bic
```

- Covariance types: `"full"` `(k,p,p)`, `"tied"` `(p,p)`, `"diag"` `(k,p)`,
  `"spherical"` `(k,)` — the shape of `covariances()` follows sklearn.
- **`collapsed()` reports EM collapse** (a component shrinking onto a
  point); `regCovar` is the exposed floor that survives it. A healthy fit
  reports `false`.
- `converged()`, `iterations()`, and `logLikelihoodHistory()` (monotone
  non-decreasing — EM's contract) are all exposed.
- Init is THIS library's seeded `KMeans` (or seeded random) — converged
  parameters match sklearn; iteration paths do not.

## AgglomerativeClustering

```cajeta
AgglomerativeClustering ag = heap AgglomerativeClustering((int64) 2);          // ward
AgglomerativeClustering av = heap AgglomerativeClustering((int64) 2, "average",
                                                          heap Manhattan());
ag.fit(x, yIgnored);
Tensor<float64> z = ag.linkageMatrix();      // (n-1, 4) scipy format
Tensor<float64> l3 = ag.cutByCount((int64) 3);   // many k from ONE fit
int64[] order = ag.leafOrder();              // dendrogram data — cajeta-chart draws
float64 c = ag.copheneticCorrelation();      // tree faithfulness
```

- **`"ward"` + a metric does not construct** — the guard is static because
  ward's update formula is Euclidean-only. Other linkages take any Metric.
- Linkages: `ward` (default) / `complete` / `average` / `single`. Single
  linkage CHAINS — the one that separates rings and moons, and the one
  that shatters on noisy bridges.

## DBSCAN

```cajeta
DBSCAN db = heap DBSCAN(0.45, (int64) 4);    // eps, minSamples (counts SELF)
db.fit(x, yIgnored);
int64 k = db.nClusters();                    // an OUTPUT
Tensor<float64> lab = db.labels();           // -1 = noise, kept as noise
boolean[] core = db.coreMask();              // border = labelled && !core
float64[] curve = DBSCAN.kDistances(x, (int64) 4);  // sorted; elbow picks eps
```

- Labels match sklearn EXACTLY, including the order-dependent border
  assignment (a border point reachable from two clusters joins whichever
  expansion pops it first — inherent to the algorithm; we replicate
  sklearn's loop rather than pretend it away).
- Failure mode to expect: continuous density between real groups → one
  merged cluster. DBSCAN separates by DENSITY GAP, not by model.

## Shared rules

- `fit(x, y)` ignores `y` (lifecycle uniformity); everything is a
  `Predictor` or `Estimator` and misuse throws `MlException`.
- Deterministic: seeded where randomized, tie-breaks documented, repeated
  fits bit-identical.
- Evaluation lives in `Metrics` — see `ml-cluster-evaluation`.
