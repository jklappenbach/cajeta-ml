---
id: ml-model-selection
applies-to: [dev.cajeta.ml.select.StratifiedKFold, dev.cajeta.ml.select.RepeatedHoldout, dev.cajeta.ml.select.GridSearch, dev.cajeta.ml.select.RandomizedSearch, dev.cajeta.ml.select.Scorers, dev.cajeta.ml.EstimatorFactory, dev.cajeta.ml.select.ForwardSelector, dev.cajeta.ml.preprocess.OneHotEncoder, dev.cajeta.ml.preprocess.OrdinalEncoder, dev.cajeta.ml.inspect.PrCurve, dev.cajeta.ml.inspect.ClassificationReport]
title: Model selection 0.8.0 — stratification, repeated holdout, grid/random search, encoders, forward selection, PR curves & reports
description: The 0.8.0 selection surface — stratified splits/folds that fail loudly, repeated holdout with per-split scores, hyperparameter search over EstimatorFactory with failure capture and Scorers metric names, categorical encoders with a loud unseen policy, greedy CV-scored forward feature selection, and the PR-curve/average-precision/classification-report trio.
---

# Model selection & reporting (0.8.0 surface)

## Stratified splits and folds — loud, not approximate

```cajeta
Tensor<float64>[] parts = Split.trainTestSplitStratified(x, y, 0.25, (uint64) 42);
StratifiedKFold sk = heap StratifiedKFold((int64) 5, true, (uint64) 11);
Tensor<int64> foldIds = sk.assignments(y);     // takes y, not n
```

- Proportions are preserved in EVERY part/fold. A class too small to
  give each part a member **fails naming the class** — never an empty
  fold. At 3% positives this is the difference between measuring and
  guessing (the suite pins the fixture).
- `RepeatedHoldout(n, frac, seed)` — n independent seeded splits;
  results carry per-split scores PLUS mean and standard deviation. Never
  report the mean alone.

## Hyperparameter search — the `EstimatorFactory` seam

```cajeta
final class MyKnnFactory implements EstimatorFactory {
    public #Predictor create(Tensor<float64> params) {
        Predictor p = heap KNeighborsClassifier((int64) params.flatGet(0));
        return #p;
    }
}
SearchResult r = GridSearch.run(heap MyKnnFactory(), dims, x, y, foldIds, "f1");
SearchResult rr = RandomizedSearch.run(heap MyKnnFactory(), dims,
    (int64) 20, (uint64) 9, x, y, foldIds, "f1");
```

- `create(params)` builds a FRESH estimator per fold — no state leaks.
- Metrics by name via `Scorers`: `accuracy` · `f1` · `rocAuc` (needs
  `predictProba`) · `r2`. Unknown names THROW — an unbalanced problem is
  never scored on accuracy silently.
- **A combination that throws is recorded and the sweep continues** —
  `SearchResult.ok(i)` / `failures()`; `best*` consider completed
  combinations only (and throw if ALL failed).
- A 1-D grid over `k` IS the K-versus-error table: iterate
  `param(i, 0)` / `1 - score(i)`.
- Same seed ⇒ identical random sweep, draw for draw.

## Encoders — categories in, tensors out, loudly

```cajeta
OneHotEncoder oh = heap OneHotEncoder();            // unseen = ERROR
OneHotEncoder oi = heap OneHotEncoder(false, true); // unseen = all-zeros row
OrdinalEncoder oe = heap OrdinalEncoder(#orderings); // caller-supplied order
```

- One-hot categories are SORTED per input column (sklearn's order);
  `dropFirst` drops the first category per column (intercept-friendly).
- The unseen-category DEFAULT is error; ignore-to-all-zeros is the
  explicit second ctor flag. Both are `Transformer`s — fit on training
  data only, so CV folds cannot leak categories (put them in the
  `Pipeline`).

## Forward feature selection — greedy, CV-scored, a Transformer

```cajeta
ForwardSelector fs = heap ForwardSelector(heap MyFactory(), params,
    /*nSelect=*/3, /*nFolds=*/3, "accuracy", (uint64) 7);
fs.fit(x, y);                 // fs.selectedCount(), fs.selectedAt(i)
Tensor<float64> xSel = fs.transform(x);   // selected columns, ascending
```

- Each round cross-validates every candidate feature added to the
  current subset and keeps the best; stops at `nSelect` or when no
  candidate STRICTLY improves. Ties break to the lowest index; folds are
  seeded — runs reproduce exactly.
- It composes: `Pipeline.of(#fs, #scaler, #model)` refits the selection
  inside each outer CV fold.

## PR curves, average precision, the report

```cajeta
PrCurve c = Metrics.precisionRecallCurve(yTrue, scores);
// points() == thresholds() + 1: sklearn's terminal (1, 0) point has NO threshold
float64 ap = Metrics.averagePrecision(yTrue, scores);
ClassificationReport rep = Metrics.classificationReport(yTrue, yPred, k);
// rep.precision(c)/recall(c)/f1(c)/support(c); macro*/weighted*; totalSupport()
```

- The curve is HOW a decision threshold gets chosen; feed the winner to
  `LogisticRegression.predict(x, t)` (see `ml-classification` for the
  `>=` vs `>` hair).
- Average precision is the step-function area (sklearn's), not a
  trapezoid.
- The report is sklearn's `output_dict` shape as a typed result — macro
  weights classes equally, weighted weights by support.
- `Metrics.confusionMatrix` is TRUE-on-rows (sklearn); much of the
  literature draws the transpose — transpose before textbook comparison.

## Hazards

- `StratifiedKFold.assignments` takes `y`; plain `KFold.assignments`
  takes `n`. Passing the wrong one is a shape error at best.
- `rocAuc` in a search demands an estimator exposing `predictProba`;
  the failure is loud, but it fails the COMBINATION — check
  `failures()` before trusting a sweep that used it.
- `ForwardSelector` can stop UNDER the target count (no-improvement
  stop) — read `selectedCount()`, don't assume `nSelect`.
