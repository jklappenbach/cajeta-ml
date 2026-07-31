#!/usr/bin/env python3
# cajeta-ml v2 Unit 2 — multinomial softmax fixtures against PINNED sklearn
# (scikit-learn==1.9.0; LogisticRegression is softmax for K>2 since the
# multi_class removal). Probabilities are parametrization-invariant — pin
# those, never raw coefficients (the intercept gauge is optimizer-path
# dependent). Dataset bit-reconstructible in cajeta.
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

print("sklearn", sklearn.__version__)

n = 30
X = np.zeros((n, 2))
y = np.zeros(n, dtype=int)
for i in range(n):
    t = i / 10.0                      # 0.0 .. 2.9
    X[i, 0] = t
    X[i, 1] = ((i * 7) % 5 - 2) / 2.0
    y[i] = 0 if t < 1.0 else (1 if t < 2.0 else 2)   # bands on x0

m = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(X, y)
P = m.predict_proba(X)
print("classes", m.classes_, "iters", m.n_iter_)
for r in (0, 7, 15, 22, 29):
    print(f"row {r}: label={m.predict(X[r:r+1])[0]} proba={[repr(v) for v in P[r]]}")
print("train accuracy", m.score(X, y))

# Binary sanity: 2-band labels, softmax == binary logistic decision
yb = (y >= 1).astype(int)
mb = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(X, yb)
Pb = mb.predict_proba(X)
for r in (0, 15, 29):
    print(f"bin row {r}: proba1={Pb[r,1]!r}")
