# cajeta-ml documentation

Classical (non-deep) machine learning for cajeta — the scikit-learn /
statsmodels role, over `Tensor<float64>` data and the stdlib's native
linear-algebra layer (`cajeta.math.linalg`: Householder QR, bidiagonal SVD,
Cholesky factor application) and special functions (`cajeta.math.stats`).

| Page | What it covers |
|---|---|
| [Guide](Guide.md) | The estimator protocol, every model and utility, API shapes |
| [Tour](Tour.md) | The runnable walkthrough (`cajeta tour`), section by section |
| [Differences from scikit-learn](DifferencesFromSklearn.md) | Where and why this library deliberately diverges |

Agent-facing **skills** ship inside the `.cja` (`skills/*.md`, indexed):
`cajeta search-skill dev.cajeta.ml` / `list-skills` / `get-skills` (or the
same over cajeta-mcp) route a coding agent to the right estimator, the
protocol contract, and the hazards — start at `ml-overview`.

Quick start:

```cajeta
LinearRegression lr = heap LinearRegression(true);
lr.fit(xTrain, yTrain);                        // Tensor<float64> (n,p), (n,)
float64 r2 = lr.score(xTest, yTest);
SummaryResult s = lr.summary(xTrain, yTrain);  // stderr, t, p-values, R², F
```

Tests are pinned against scikit-learn 1.9.0 / scipy 1.18.0; the inference
statistics against the classical closed-form OLS formulas (exact
t-distribution p-values).
