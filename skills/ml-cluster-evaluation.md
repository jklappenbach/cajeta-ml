---
id: ml-cluster-evaluation
applies-to: [dev.cajeta.ml.Metrics]
title: Cluster evaluation — silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI, the elbow
description: The unsupervised half of Metrics — three internal judges that can DISAGREE (offering three is the point), two external benchmarks against ground truth, the k-sweep elbow, and the degenerate cases that throw instead of misleading.
---

# Cluster evaluation

All statics on `Metrics`, over `(n, p)` data and `(n,)` float64 labels
(any distinct values work; DBSCAN's `-1` participates as its own group).

## Internal judges (no ground truth)

```cajeta
float64 s  = Metrics.silhouetteScore(x, labels);       // higher better, [-1, 1]
Tensor<float64> per = Metrics.silhouetteSamples(x, labels);  // for silhouette PLOTS
float64 db = Metrics.daviesBouldin(x, labels);         // LOWER better
float64 ch = Metrics.calinskiHarabasz(x, labels);      // higher better
```

- **Silhouette on 1 cluster, or n clusters, THROWS** — it is undefined
  there, and an undefined number returned anyway is how bad k gets chosen.
  Singleton clusters score 0 (the sklearn convention).
- **The three can disagree**, and that is why all three exist: on two
  close tight clusters plus a far one, silhouette AND Davies-Bouldin
  prefer merging the close pair while Calinski-Harabasz prefers the true
  split (the suite pins this example). Read them as three priors —
  separation-vs-cohesion (silhouette), worst-pair (DB), variance ratio
  (CH) — not as one number three ways.

## External benchmarks (ground truth known)

```cajeta
float64 ari = Metrics.adjustedRand(yTrue, yPred);  // 1 identical, ~0 chance
float64 nmi = Metrics.nmi(yTrue, yPred);           // arithmetic normalization
```

Both are label-permutation invariant — renaming clusters changes nothing.
ARI is chance-corrected pair counting; NMI is information-theoretic
(sklearn's default `arithmetic` normalization, no knob).

## Choosing k

```cajeta
float64[] inertias = Metrics.kmeansElbow(x, (int64) 8, seed);  // k = 1..8
```

The bend in the curve is the candidate k; confirm with silhouette/BIC
(`GaussianMixture.bic`) rather than trusting one statistic — see the
disagreement note above.
