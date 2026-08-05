#!/usr/bin/env python3
# ml-classification-gaps U4 — cost-aware + sparse logistic pins against
# PINNED sklearn 1.9.0. L1/elastic-net oracle is saga (liblinear penalizes
# the intercept; saga matches spec §6.6's unpenalized intercept).
# Run: /home/julian/code/ml/venv-sklearn-ref/bin/python gen_logistic_weights.py
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

print("sklearn", sklearn.__version__)

# Binary fixture, 20 x 2, overlapping, bit-reconstructible.
n = 20
X = np.zeros((n, 2))
y = np.zeros(n, dtype=int)
for i in range(n):
    X[i, 0] = ((i * 7) % 13) * 0.3 - 1.8
    X[i, 1] = ((i * 5) % 11) * 0.35 - 1.6
    s = X[i, 0] + 0.8 * X[i, 1] + (((i * 3) % 7) - 3) * 0.45
    y[i] = 1 if s > 0 else 0
print("y:", y.tolist())

def dump(tag, arr):
    a = np.asarray(arr).ravel()
    print(tag, [repr(v) for v in a])

# Explicit class weights (binary).
cw = LogisticRegression(C=1.0, class_weight={0: 1.0, 1: 3.0}, tol=1e-12,
                        max_iter=10000).fit(X, y)
dump("cw coef:", cw.coef_)
dump("cw icept:", cw.intercept_)
dump("cw proba3:", cw.predict_proba(X[:3])[:, 1])

# Balanced on an imbalanced fixture (same X; y2 has 4 positives).
y2 = np.zeros(n, dtype=int)
for i in (3, 8, 15, 19):
    y2[i] = 1
bal = LogisticRegression(C=1.0, class_weight="balanced", tol=1e-12,
                         max_iter=10000).fit(X, y2)
dump("bal coef:", bal.coef_)
dump("bal icept:", bal.intercept_)
exp = LogisticRegression(C=1.0, class_weight={0: 20/(2*16), 1: 20/(2*4)},
                         tol=1e-12, max_iter=10000).fit(X, y2)
dump("bal explicit coef:", exp.coef_)   # must equal bal

# Sample weights composing multiplicatively with class weights.
sw = np.array([1.0 + (i % 4) * 0.5 for i in range(n)])
comp = LogisticRegression(C=1.0, class_weight={0: 1.0, 1: 2.0}, tol=1e-12,
                          max_iter=10000).fit(X, y, sample_weight=sw)
dump("comp coef:", comp.coef_)
dump("comp icept:", comp.intercept_)

# Multiclass (multinomial) class weights.
y3 = np.zeros(n, dtype=int)
for i in range(n):
    y3[i] = 0 if X[i, 0] + 0.2 * ((i % 5) - 2) < -0.6 else (
        1 if X[i, 0] < 0.7 else 2)
print("y3:", y3.tolist())
mc = LogisticRegression(C=1.0, class_weight={0: 1.0, 1: 2.0, 2: 3.0},
                        tol=1e-12, max_iter=10000).fit(X, y3)
dump("mc proba0:", mc.predict_proba(X[:2]))

# L1 by saga at two C values.
for c in (5.0, 1.5):
    l1 = LogisticRegression(C=c, l1_ratio=1.0, solver="saga", tol=1e-12,
                            max_iter=500000).fit(X, y)
    dump(f"l1 C={c} coef:", l1.coef_)
    dump(f"l1 C={c} icept:", l1.intercept_)

# A wider fixture so L1 produces exact zeros at moderate C.
Xw = np.zeros((n, 5))
for i in range(n):
    for j in range(5):
        Xw[i, j] = ((i * (3 + j) + j * 2) % 9) * 0.3 - 1.2
l1w = LogisticRegression(C=0.8, l1_ratio=1.0, solver="saga", tol=1e-12,
                         max_iter=500000).fit(Xw, y)
dump("l1 wide coef:", l1w.coef_)
dump("l1 wide icept:", l1w.intercept_)

# Elastic-net ratio 0.5 (saga).
en = LogisticRegression(C=0.5, solver="saga", l1_ratio=0.5, tol=1e-12,
                        max_iter=500000).fit(Xw, y)
dump("enet coef:", en.coef_)
dump("enet icept:", en.intercept_)

# Pure-L2 reference for the ratio=0 self-consistency check.
l2 = LogisticRegression(C=0.5, tol=1e-12, max_iter=10000).fit(Xw, y)
dump("l2 coef:", l2.coef_)
dump("l2 icept:", l2.intercept_)
