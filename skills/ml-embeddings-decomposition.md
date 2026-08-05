---
id: ml-embeddings-decomposition
applies-to: [dev.cajeta.ml.TSNE, dev.cajeta.ml.MDS, dev.cajeta.ml.Isomap, dev.cajeta.ml.LLE, dev.cajeta.ml.SpectralEmbedding, dev.cajeta.ml.FastICA, dev.cajeta.ml.NMF, dev.cajeta.ml.FactorAnalysis]
title: Embeddings & decomposition — t-SNE, MDS, Isomap, LLE, SpectralEmbedding, FastICA, NMF, FactorAnalysis
description: The manifold/embedding family with its honesty rules (who can transform unseen points and who only pretends), the linear decompositions beyond PCA, and the indeterminacies (sign, rotation, permutation, scale) that make raw-coordinate comparison a bug.
---

# Embeddings & decomposition

## The honesty rules (learn these first)

- **Who can embed UNSEEN points**: `Isomap` and `LLE` have `transform`
  (real out-of-sample extensions; both are `Transformer`s). `TSNE`, `MDS`,
  and `SpectralEmbedding` are **`fitTransform`-only** — no `transform`
  exists on them and they do not implement `Transformer`, because offering
  one would be a lie.
- **Indeterminacy is the rule, not a bug** (§12.9 doctrine): eigenvector
  SIGN (all spectral methods), ROTATION (MDS, FactorAnalysis loadings),
  PERMUTATION + SCALE (ICA sources, NMF factors). Compare eigenvalues,
  distance structure, |correlation|, `ΛΛᵀ` — never raw coordinates.
- **Exact and dense**: everything here is O(n²)–O(n³). `TSNE` REJECTS
  n > 2000 with instructions (Barnes-Hut is a recorded deferral).

## t-SNE

```cajeta
TSNE ts = heap TSNE((int64) 2, /*perplexity*/ 30.0, seed);
Tensor<float64> e = ts.fitTransform(x);   // (n, 2)
float64 kl = ts.klDivergence();           // the minimized objective (stdlib KL)
```

- **Between-cluster distances in the picture are NOT meaningful**, and
  perplexity materially changes it — run several perplexities before
  believing anything. The most over-interpreted plot in data science.
- Seed required (random init); `3·perplexity >= n` is rejected.

## MDS

```cajeta
MDS m = heap MDS((int64) 2, seed);                         // metric SMACOF
MDS nm = heap MDS((int64) 2, seed, "nonmetric", (int64) 8,
                  (int64) 500, 0.000000001);               // variant is NAMED
Tensor<float64> e = m.fitTransform(x);
Tensor<float64> e2 = m.fitTransformDissimilarity(d);        // any square pdist result
float64 s = m.stress();   // metric: raw Σ(δ−d)²; nonmetric: normalized stress-1
```

The precomputed door is why MDS composes with `cajeta.math.distance`
instead of owning metrics. `nInit` restarts keep the best; stress is the
fit-quality number — report it with the plot.

## The spectral trio

```cajeta
Isomap iso = heap Isomap(/*nNeighbors*/ (int64) 8, (int64) 2);
Tensor<float64> e = iso.fitTransform(x, yIgnored);
Tensor<float64> u = iso.transform(unseen);      // real out-of-sample; exact on training rows
float64[] lam = iso.kernelEigenvalues();        // the sign-free parity quantity

LLE lle = heap LLE((int64) 8, (int64) 2);       // reg 1e-3 default
SpectralEmbedding sk = heap SpectralEmbedding((int64) 2, (int64) 8);   // kNN affinity
SpectralEmbedding sr = heap SpectralEmbedding((int64) 2, /*gamma*/ 0.5); // RBF affinity
```

- **A disconnected k-NN graph throws, naming `nNeighbors`** (Isomap) —
  the classic silent-wrongness of manifold methods, made loud.
- SpectralEmbedding drops the trivial `λ≈0` eigenvector (documented);
  `eigenvalues()` returns the kept non-trivial ones.

## Linear decomposition beyond PCA

```cajeta
FastICA ica = heap FastICA((int64) 3, seed);          // logcosh; also "exp"
Tensor<float64> sources = ica.fitTransform(x, yIgnored);  // up to sign/perm/scale
// NON-CONVERGENCE THROWS — an unconverged unmixing is never returned.

NMF nmf = heap NMF((int64) 2);                        // NNDSVD init — deterministic, no seed
nmf.fit(xNonNegative, yIgnored);                      // negative input rejected NAMING the entry
Tensor<float64> h = nmf.components();                 // (k, p), non-negative
float64 err = nmf.reconstructionError();              // Frobenius ‖X − WH‖

FactorAnalysis fa = heap FactorAnalysis((int64) 2);   // EM, per-feature noise
fa.fit(x, yIgnored);
Tensor<float64> psi = fa.noiseVariances();            // (p,) — FA's reason to exist
```

- **PCA vs FA**: PCA assumes isotropic noise and mixes per-feature noise
  into the components; FA models each feature's noise separately. When
  features have different error scales, FA is the right model.
- FA loadings are rotation-indeterminate — compare `loadings·loadingsᵀ`
  and `noiseVariances`, never columns. `logLikelihoodHistory()` is
  monotone (EM's contract, asserted in the suite).
- ICA/NMF have `inverseTransform` (back to feature space); FA does not
  (sklearn has none either).
