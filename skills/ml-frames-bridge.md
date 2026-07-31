---
id: ml-frames-bridge
applies-to: [dev.cajeta.ml.Frames, dev.cajeta.ml.Design]
title: Frames.design — the Table<R> → design-matrix bridge (two lines from dataframe to fit)
description: The explicit, auditable bridge from any nucleo Table<R> (Parquet/Arrow/CSV/in-memory) to any Predictor's fit — selection rules, the Design record of what was selected, every loud failure, and why there is no DataFrame duck-typing.
---

# Frames.design — dataframe → estimator, explicitly

```cajeta
import cajeta.nucleo.frame.Table;
import dev.cajeta.ml.Design;
import dev.cajeta.ml.Frames;

Design d = Frames.design<Tick>(t, "price");   // t: Table<Tick>
est.fit(d.x, d.y);                            // ANY Predictor — OLS, Ridge, XGBRegressor…
```

`Design` is three public fields: `x (n, p)` float64, `y (n,)` float64,
and `featureNames` (`String[]`, one per x column, in order) — the record
of exactly which frame column became which coefficient, so
`summary()` output can be labeled and the selection audited.

## Selection rules

- **Default overload** `design<R>(t, target)`: features = the
  **float64-physical** columns of R's schema, in schema order, minus the
  target. `int64` / `Instant` / `Utf8` columns are SKIPPED — never
  silently coerced (engineer them into float64 columns in the frame
  first).
- **Explicit overload** `design<R>(t, features, target)`: exactly the
  named columns, in the order you name them (subset + reorder).
- Rows come through `Table`'s dynamic reads — the table must be
  materialized (call `.collect()` on a lazy handle first).

## Loud failures (`MlException`, nothing silent)

| Input | Result |
|---|---|
| unknown target / feature name | throws, names the column |
| non-float64 target or explicit feature | throws (no coercion) |
| target also listed as a feature | throws |
| zero features selected | throws |
| nullable selected column | throws — fill/drop nulls IN THE FRAME first (`fillNull` / `dropNulls`); imputation belongs to the frame, not the bridge |

## Why a bridge instead of `fit(Table<R>)`

sklearn accepts a DataFrame anywhere and infers feature columns
implicitly; that implicitness is exactly what hides wrong-column bugs.
One explicit seam (this bridge) composes with EVERY estimator — including
other libraries' conformers (`XGBRegressor` fits from the same `Design`,
bit-identical to its tensor fit) — without N per-class overloads.

## Hazards

- Records used as `Table<R>` schemas: co-locate the record with heavy
  synthesized use if you must support pre-v0.13.0 toolchains (the
  cross-file materialization hazards are fixed from v0.13.0).
- `d.x` is a fresh copy — mutating the frame afterward does not change a
  Design already built (rebuild it).
