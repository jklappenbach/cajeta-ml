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

# ---------- U2: LDA / QDA pins ----------
# Query points, bit-reconstructible: Q[i,j] = ((i*3 + j*2) % 7) * 0.55
Q = np.zeros((4, p))
for i in range(4):
    for j in range(p):
        Q[i, j] = ((i * 3 + j * 2) % 7) * 0.55

def dump(tag, arr):
    a = np.asarray(arr)
    if a.ndim == 1:
        print(tag, [repr(v) for v in a])
    else:
        print(tag)
        for row in a:
            print("  ", [repr(v) for v in row])

# svd solver (default), 3-class
svd3 = LinearDiscriminantAnalysis().fit(X, y)
dump("lda svd predict:", svd3.predict(Q).astype(float))
dump("lda svd decision:", svd3.decision_function(Q))
dump("lda svd proba:", svd3.predict_proba(Q))
dump("lda svd transform:", svd3.transform(X[:3]))
dump("lda svd evr:", svd3.explained_variance_ratio_)

# explicit priors
pri = LinearDiscriminantAnalysis(priors=[0.5, 0.25, 0.25]).fit(X, y)
dump("lda priors decision0:", pri.decision_function(Q[:1]))

# binary (classes 0/1 only): single decision column
Xb, yb = X[:20], y[:20]
svdb = LinearDiscriminantAnalysis().fit(Xb, yb)
dump("lda bin decision:", svdb.decision_function(Q))
dump("lda bin predict:", svdb.predict(Q).astype(float))
dump("lda bin proba1:", svdb.predict_proba(Q)[:, 1])

# p > n, svd never forms the covariance: n=8, p=12, K=2
n2, p2 = 8, 12
Xw = np.zeros((n2, p2))
yw = np.array([0, 0, 0, 0, 1, 1, 1, 1])
for i in range(n2):
    for j in range(p2):
        Xw[i, j] = ((i * 5 + j * 3) % 13) * 0.21 + yw[i] * 0.9 * ((j % 3) - 1)
Qw = np.zeros((2, p2))
for i in range(2):
    for j in range(p2):
        Qw[i, j] = ((i * 4 + j * 5) % 9) * 0.33
wide = LinearDiscriminantAnalysis().fit(Xw, yw)
dump("lda wide decision:", wide.decision_function(Qw))
dump("lda wide predict:", wide.predict(Qw).astype(float))

# lsqr + shrinkage 0.3; eigen + shrinkage 0.3; lsqr auto
lsqr = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=0.3).fit(X, y)
dump("lda lsqr coef:", lsqr.coef_)
dump("lda lsqr icept:", lsqr.intercept_)
eig = LinearDiscriminantAnalysis(solver="eigen", shrinkage=0.3).fit(X, y)
dump("lda eigen coef:", eig.coef_)
dump("lda eigen icept:", eig.intercept_)
dump("lda eigen evr:", eig.explained_variance_ratio_)
dump("lda eigen transform:", eig.transform(X[:2]))
auto = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
dump("lda auto coef:", auto.coef_)
dump("lda auto icept:", auto.intercept_)

# QDA
q3 = QuadraticDiscriminantAnalysis().fit(X, y)
dump("qda decision:", q3.decision_function(Q))
dump("qda predict:", q3.predict(Q).astype(float))
dump("qda proba:", q3.predict_proba(Q))
qreg = QuadraticDiscriminantAnalysis(reg_param=0.1).fit(X, y)
dump("qda reg decision:", qreg.decision_function(Q))

# --- U1: Gaussian log-density via the pooled covariance ---
from scipy.stats import multivariate_normal
mvn = multivariate_normal(mean=lda.means_[0], cov=lda.covariance_)
print("logpdf:", [repr(v) for v in mvn.logpdf(X[:3])])

# --- U1: log-sum-exp rows incl. extreme values ---
rows = np.array([[0.1, 0.2, 0.3],
                 [-1000.0, -1000.5, -999.5],
                 [5.0, -1000.0, 4.0]])
print("logsumexp:", [repr(v) for v in logsumexp(rows, axis=1)])
