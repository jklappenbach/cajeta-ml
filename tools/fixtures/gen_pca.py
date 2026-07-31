#!/usr/bin/env python3
# cajeta-ml v2 Unit 6 — PCA fixture against PINNED sklearn 1.9.0
# (full SVD solver). Bit-reconstructible 3-feature data with a dominant
# direction.
import numpy as np
import sklearn
from sklearn.decomposition import PCA

print("sklearn", sklearn.__version__)

n = 16
X = np.zeros((n, 3))
for i in range(n):
    t = i / 4.0
    X[i, 0] = t
    X[i, 1] = 2.0 * t + ((i * 3) % 5 - 2) * 0.05
    X[i, 2] = -t + ((i * 7) % 3 - 1) * 0.1

m = PCA(n_components=2, svd_solver="full").fit(X)
print("components:")
for r in m.components_:
    print("  ", [repr(v) for v in r])
print("explained_variance", [repr(v) for v in m.explained_variance_])
print("explained_variance_ratio", [repr(v) for v in m.explained_variance_ratio_])
T = m.transform(X)
for r in (0, 7, 15):
    print(f"transform row {r}:", [repr(v) for v in T[r]])
R = m.inverse_transform(T)
print("inverse max abs err vs X:", repr(float(np.max(np.abs(R - X)))))
