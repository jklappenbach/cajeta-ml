#!/usr/bin/env python3
# ml-classification-gaps U8 — PR curve, average precision, classification
# report, forward selection pins against PINNED sklearn 1.9.0.
# Run: /home/julian/code/ml/venv-sklearn-ref/bin/python gen_reporting.py
import numpy as np
import sklearn
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
                             precision_recall_curve)

print("sklearn", sklearn.__version__)
assert sklearn.__version__ == "1.9.0", (
    f"fixtures are pinned to sklearn 1.9.0, not {sklearn.__version__}")

# Binary scores fixture: 12 rows, deterministic scores with one tie pair.
y = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
s = np.array([0.15, 0.85, 0.60, 0.35, 0.70, 0.20, 0.60, 0.72, 0.90,
              0.10, 0.65, 0.75])
prec, rec, thr = precision_recall_curve(y, s)
print("prec:", [repr(v) for v in prec])
print("rec:", [repr(v) for v in rec])
print("thr:", [repr(v) for v in thr])
print("ap:", repr(average_precision_score(y, s)))

# 3-class report fixture.
yt = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
yp = np.array([0, 0, 1, 2, 1, 1, 0, 2, 2, 2, 1, 2])
rep = classification_report(yt, yp, output_dict=True)
for k in ("0", "1", "2", "macro avg", "weighted avg"):
    d = rep[k]
    print(f"rep {k}:", repr(d["precision"]), repr(d["recall"]),
          repr(d["f1-score"]), repr(d["support"]))

# Forward selection: features 1 and 3 informative, rest lattice noise.
n = 30
X = np.zeros((n, 5))
yb = np.zeros(n, dtype=int)
for i in range(n):
    for j in range(5):
        X[i, j] = (((i * (2 + j) + j * 3) % 11) * 0.2 - 1.0) * 0.01
    X[i, 1] = ((3 * i + 5) % 13) * 0.3 - 1.8
    X[i, 3] = ((7 * i + 2) % 13) * 0.3 - 1.8
    yb[i] = 1 if X[i, 1] - X[i, 3] > 0 else 0
print("yb:", yb.tolist())
sfs = SequentialFeatureSelector(
    LogisticRegression(max_iter=2000), n_features_to_select=2,
    direction="forward", cv=3).fit(X, yb)
print("sfs support:", np.where(sfs.get_support())[0].tolist())
