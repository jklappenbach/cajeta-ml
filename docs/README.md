# cajeta-ml documentation

Machine learning for cajeta, in two halves:

- **Classical** (the scikit-learn / statsmodels role) over `Tensor<float64>`,
  built on the stdlib's native linear algebra (`cajeta.math.linalg`:
  Householder QR, bidiagonal SVD, Cholesky factor application) and special
  functions (`cajeta.math.stats`).
- **Neural** (the torch role) over `Tensor<float32>` — a runtime tensor tape,
  modules and layers, optimizers, two trainers, and torch-checkpoint interop.

The precision boundary is crossed exactly once, at the estimator adapter.

| Page | What it covers |
|---|---|
| [Guide](Guide.md) | The estimator protocol, every model and utility, API shapes |
| [Tour](Tour.md) | The runnable walkthrough (`cajeta tour`), section by section |
| [Differences from scikit-learn](DifferencesFromSklearn.md) | Where and why the classical half deliberately diverges |
| [Differences from PyTorch](DifferencesFromTorch.md) | The same, for the neural half — plus what is honestly not implemented |

Agent-facing **skills** ship inside the `.cja` (`skills/*.md`, indexed):
`cajeta search-skill dev.cajeta.ml` / `list-skills` / `get-skills` (or the
same over cajeta-mcp) route a coding agent to the right estimator, the
protocol contract, and the hazards — start at `ml-overview`. Neural work is
covered by `ml-grad`, `ml-nn-modules`, `ml-training` and
`ml-checkpoints-lora`.

Quick start:

```cajeta
LinearRegression lr = heap LinearRegression(true);
lr.fit(xTrain, yTrain);                        // Tensor<float64> (n,p), (n,)
float64 r2 = lr.score(xTest, yTest);
SummaryResult s = lr.summary(xTrain, yTrain);  // stderr, t, p-values, R², F
```

Tests are pinned against scikit-learn 1.9.0 / scipy 1.18.0 (classical) and
torch 2.13.0+cpu (neural) — optimizer trajectories and training curves step by
step, not merely at convergence. The inference statistics are pinned against
the classical closed-form OLS formulas (exact t-distribution p-values).
