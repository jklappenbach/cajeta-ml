#!/usr/bin/env python3
# cajeta-ml v2 Unit 5 — kNN fixtures against PINNED sklearn 1.9.0.
# Tie-free 2-D points (irrational-ish spacing via integer grids that never
# equalize distances at the query points), bit-reconstructible in cajeta.
import numpy as np
import sklearn
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

print("sklearn", sklearn.__version__)

n = 12
X = np.zeros((n, 2))
yr = np.zeros(n)
yc = np.zeros(n, dtype=int)
for i in range(n):
    X[i, 0] = i * 0.7
    X[i, 1] = ((i * 5) % 7) * 0.31
    yr[i] = 2.0 * X[i, 0] - X[i, 1] + 0.25
    yc[i] = 0 if i < 6 else 1

Q = np.array([[1.0, 0.5], [5.3, 1.2], [7.7, 0.1]])

for w in ("uniform", "distance"):
    r = KNeighborsRegressor(n_neighbors=3, weights=w).fit(X, yr)
    print(f"reg {w}:", [repr(v) for v in r.predict(Q)])

c = KNeighborsClassifier(n_neighbors=3).fit(X, yc)
print("cls labels:", c.predict(Q).tolist())
print("cls proba:", [[repr(v) for v in row] for row in c.predict_proba(Q)])
