#!/usr/bin/env python3
# cajeta-ml v2 Unit 1 — Lasso/ElasticNet fixtures against PINNED sklearn.
# Run with the pinned venv: /home/julian/code/ml/venv-sklearn-ref/bin/python
# (scikit-learn==1.9.0). The dataset is bit-reconstructible in cajeta:
# integer-derived float64 formulas only.
import numpy as np
import sklearn
from sklearn.linear_model import Lasso, ElasticNet

print("sklearn", sklearn.__version__)

n, p = 20, 3
X = np.zeros((n, p))
y = np.zeros(n)
for i in range(n):
    t = i / 10.0                     # 0.0 .. 1.9
    X[i, 0] = t
    X[i, 1] = ((i * 7) % 5 - 2) / 2.0   # junk-ish grid, mean ~0
    X[i, 2] = t * t
    # target: 2 + 1.5*x0 - 0.8*x2, deterministic +-0.01 ripple
    y[i] = 2.0 + 1.5 * X[i, 0] - 0.8 * X[i, 2] + (0.01 if i % 2 == 0 else -0.01)

def dump(name, m):
    c = m.coef_
    print(f"{name}: intercept={m.intercept_!r} coef={[repr(v) for v in c]}"
          f" iters={m.n_iter_}")

for alpha in (0.01, 0.5):
    m = Lasso(alpha=alpha, tol=1e-10, max_iter=100000).fit(X, y)
    dump(f"lasso a={alpha}", m)

m = ElasticNet(alpha=0.05, l1_ratio=0.5, tol=1e-10, max_iter=100000).fit(X, y)
dump("enet a=0.05 l1r=0.5", m)
m = ElasticNet(alpha=0.01, l1_ratio=1.0, tol=1e-10, max_iter=100000).fit(X, y)
dump("enet a=0.01 l1r=1.0 (== lasso)", m)
