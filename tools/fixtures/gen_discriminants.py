#!/usr/bin/env python3
# ml-classification-gaps U1/U2 — shared Gaussian machinery + LDA/QDA pins
# against PINNED sklearn 1.9.0.
# Run: /home/julian/code/ml/venv-sklearn-ref/bin/python gen_discriminants.py
import numpy as np
import sklearn
from scipy.special import logsumexp
from sklearn.covariance import ledoit_wolf
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)

print("sklearn", sklearn.__version__)

# 30 x 3, three classes of 10, bit-reconstructible in cajeta:
#   X[i,j] = ((i*7 + j*5) % 11) * 0.3 + c * shift[j],  c = i // 10
n, p, K = 30, 3, 3
shift = [1.25, -0.75, 0.5]
X = np.zeros((n, p))
y = np.zeros(n, dtype=int)
for i in range(n):
    c = i // 10
    y[i] = c
    for j in range(p):
        X[i, j] = ((i * 7 + j * 5) % 11) * 0.3 + c * shift[j]

# --- U1: pooled covariance, sklearn LDA convention (priors-weighted ML) ---
lda = LinearDiscriminantAnalysis(solver="lsqr", store_covariance=True).fit(X, y)
print("pooled cov:")
for row in lda.covariance_:
    print("  ", [repr(v) for v in row])
print("class means:")
for row in lda.means_:
    print("  ", [repr(v) for v in row])

# --- U1: per-class covariance, sklearn QDA convention (divisor n_k - 1) ---
qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)
print("class 0 cov:")
for row in qda.covariance_[0]:
    print("  ", [repr(v) for v in row])
print("class 2 cov:")
for row in qda.covariance_[2]:
    print("  ", [repr(v) for v in row])

# --- U1: Ledoit-Wolf on the pooled sample ---
lw_cov, lw_shrink = ledoit_wolf(X)
print("lw shrinkage:", repr(lw_shrink))
print("lw cov:")
for row in lw_cov:
    print("  ", [repr(v) for v in row])

# --- U1: Gaussian log-density via the pooled covariance ---
from scipy.stats import multivariate_normal
mvn = multivariate_normal(mean=lda.means_[0], cov=lda.covariance_)
print("logpdf:", [repr(v) for v in mvn.logpdf(X[:3])])

# --- U1: log-sum-exp rows incl. extreme values ---
rows = np.array([[0.1, 0.2, 0.3],
                 [-1000.0, -1000.5, -999.5],
                 [5.0, -1000.0, 4.0]])
print("logsumexp:", [repr(v) for v in logsumexp(rows, axis=1)])
