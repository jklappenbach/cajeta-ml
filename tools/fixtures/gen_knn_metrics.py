#!/usr/bin/env python3
# ml-classification-gaps U3 — k-NN metric + distance-weighting pins
# against PINNED sklearn 1.9.0.
# Run: /home/julian/code/ml/venv-sklearn-ref/bin/python gen_knn_metrics.py
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

print("sklearn", sklearn.__version__)

# The gen_knn.py fixture: tie-free 12 x 2.
n = 12
X = np.zeros((n, 2))
yr = np.zeros(n)
yc = np.zeros(n, dtype=int)
for i in range(n):
    X[i, 0] = i * 0.7
    X[i, 1] = ((i * 5) % 7) * 0.31
    yr[i] = 2.0 * X[i, 0] - X[i, 1] + 0.25
    yc[i] = 0 if i < 6 else 1

Q = np.array([[3.9, 0.8], [4.3, 1.4], [3.6, 0.2]])
Q2 = np.array([[1.0, 0.5], [5.3, 1.2], [7.7, 0.1]])

for met, kw in [("manhattan", {}), ("chebyshev", {}),
                ("minkowski", {"p": 3})]:
    c = KNeighborsClassifier(n_neighbors=3, metric=met, **kw).fit(X, yc)
    print(f"cls {met} labels:", c.predict(Q).tolist())
    print(f"cls {met} proba:", [[repr(v) for v in r] for r in c.predict_proba(Q)])
    r = KNeighborsRegressor(n_neighbors=3, metric=met, **kw).fit(X, yr)
    print(f"reg {met}:", [repr(v) for v in r.predict(Q2)])

# Distance weighting on the CLASSIFIER (euclidean + manhattan).
for met in ("euclidean", "manhattan"):
    c = KNeighborsClassifier(n_neighbors=3, weights="distance",
                             metric=met).fit(X, yc)
    print(f"cls dw {met} labels:", c.predict(Q).tolist())
    print(f"cls dw {met} proba:",
          [[repr(v) for v in r] for r in c.predict_proba(Q)])

# Exact coincidence under distance weights: the match takes all the weight.
Qz = np.array([X[7], [1.0, 0.5]])
c = KNeighborsClassifier(n_neighbors=3, weights="distance").fit(X, yc)
print("cls dw coincide proba:",
      [[repr(v) for v in r] for r in c.predict_proba(Qz)])
