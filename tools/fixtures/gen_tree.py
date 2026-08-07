#!/usr/bin/env python3
# ml-trees-ensembles U1 — decision-tree pins against PINNED sklearn 1.9.0.
# Run: /home/julian/code/ml/venv-sklearn-ref/bin/python gen_tree.py
#
# Fixtures are TIE-FREE ACROSS FEATURES by construction (integer-grid values
# with distinct impurity decreases), because sklearn breaks cross-feature ties
# by its RNG's feature visit order (_splitter.pyx node_split_best: Fisher-Yates
# feature draw + strict `>`), which is not a rule worth reproducing. The
# WITHIN-feature rule IS deterministic and pinned here: positions scan in
# ascending sorted order and strict `>` keeps the FIRST (lowest threshold);
# the `tie` fixture below is constructed to exercise exactly that.
# Thresholds are `v[p_prev]/2 + v[p]/2` (sum of halves).
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

print("sklearn", sklearn.__version__)

def dump(name, t):
    tr = t.tree_
    print(f"--- {name}: depth={t.get_depth()} leaves={t.get_n_leaves()} nodes={tr.node_count}")
    for i in range(tr.node_count):
        if tr.children_left[i] == -1:
            print(f"  node {i}: LEAF value={[repr(v) for v in tr.value[i].ravel()]}")
        else:
            print(f"  node {i}: f={tr.feature[i]} thr={tr.threshold[i]!r} "
                  f"L={tr.children_left[i]} R={tr.children_right[i]}")

# ---- cls: 12 rows, 2 features, 3 classes; tie-free integer-ish grid ----
n = 12
X = np.zeros((n, 2))
y = np.zeros(n, dtype=int)
for i in range(n):
    X[i, 0] = float(i)                      # monotone feature
    X[i, 1] = float((i * 5) % 12)           # scrambled feature
y[:4] = 0; y[4:8] = 1; y[8:] = 2
# NO perturbation: a 2-sample impure node ties across EVERY feature that
# separates the pair, which sklearn resolves by RNG visit order — the exact
# cross-feature-tie territory these fixtures must avoid. Contiguous class
# blocks on f0 keep every impure node >= 4 samples with a strictly best
# split. (The root DOES tie WITHIN f0 — 3.5 vs 7.5 by symmetry — pinning
# the ascending-scan first-kept rule on a real fixture.)

for crit in ("gini", "entropy"):
    c = DecisionTreeClassifier(criterion=crit, random_state=1).fit(X, y)
    dump(f"cls-{crit}", c)
    Q = np.array([[1.0, 3.0], [5.1, 7.0], [10.5, 2.0], [2.7, 11.0]])
    print("  labels:", c.predict(Q).tolist())
    print("  proba:", [[repr(v) for v in row] for row in c.predict_proba(Q)])

# ---- pure-node stop: one feature separates perfectly ----
Xp = np.array([[0.0], [1.0], [2.0], [3.0], [10.0], [11.0], [12.0], [13.0]])
yp = np.array([0, 0, 0, 0, 1, 1, 1, 1])
p = DecisionTreeClassifier(criterion="gini", random_state=1).fit(Xp, yp)
dump("cls-pure", p)

# ---- within-feature tie: symmetric labels -> two equal-decrease positions ----
# One feature; y = 0 0 1 1 0 0. Splitting at 1.5 (pos 2) and at 3.5 (pos 4)
# both isolate the middle pair partially with EQUAL impurity decrease;
# sklearn keeps the FIRST in ascending scan (threshold 1.5).
Xt = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
yt = np.array([0, 0, 1, 1, 0, 0])
t = DecisionTreeClassifier(criterion="gini", random_state=1).fit(Xt, yt)
dump("cls-tie", t)

# ---- reg: leaf means + squared-error splits ----
Xr = np.zeros((n, 2))
yr = np.zeros(n)
for i in range(n):
    Xr[i, 0] = float(i)
    Xr[i, 1] = float((i * 7) % 12)
    yr[i] = 3.0 * Xr[i, 0] + 0.5 * Xr[i, 1] - 2.0
r = DecisionTreeRegressor(criterion="squared_error", random_state=1, max_depth=3).fit(Xr, yr)
dump("reg-d3", r)
Qr = np.array([[1.0, 3.0], [5.1, 7.0], [10.5, 2.0]])
print("  pred:", [repr(v) for v in r.predict(Qr)])

# ---- export_text as the rule-dump reference shape (not asserted verbatim) ----
print("--- export (cls-pure):")
print(export_text(p, feature_names=["f0"]))

# ---- impure leaves: depth-1 on the 3-class data -> fractional proba ----
c1 = DecisionTreeClassifier(criterion="gini", random_state=1, max_depth=1).fit(X, y)
dump("cls-d1", c1)
Q1 = np.array([[1.0, 3.0], [10.5, 2.0]])
print("  labels:", c1.predict(Q1).tolist())
print("  proba:", [[repr(v) for v in row] for row in c1.predict_proba(Q1)])

# ---- U2: regression criteria (absolute_error -> median leaves, poisson) ----
Xa = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0]])
ya = np.array([1.0, 2.0, 2.0, 100.0, 40.0, 41.0, 43.0, 45.0])  # outlier at row 3
for crit in ("squared_error", "absolute_error"):
    m = DecisionTreeRegressor(criterion=crit, random_state=1, max_depth=1).fit(Xa, ya)
    dump(f"reg-{crit}-d1", m)
    print("  pred:", [repr(v) for v in m.predict(np.array([[1.5], [6.0]]))])

yp2 = np.array([0.0, 1.0, 1.0, 2.0, 8.0, 9.0, 11.0, 14.0])     # counts
mp = DecisionTreeRegressor(criterion="poisson", random_state=1, max_depth=1).fit(Xa, yp2)
dump("reg-poisson-d1", mp)
print("  pred:", [repr(v) for v in mp.predict(np.array([[1.5], [6.0]]))])
# deeper AE tree: median leaves at depth 2
m2 = DecisionTreeRegressor(criterion="absolute_error", random_state=1, max_depth=2).fit(Xa, ya)
dump("reg-absolute_error-d2", m2)
print("  pred:", [repr(v) for v in m2.predict(np.array([[0.5], [3.0], [4.5], [6.5]]))])

# ---- U3: pruning and regularization ----
# One noisy 1-D dataset drives all the pruning pins.
Xu = np.arange(16.0).reshape(-1, 1)
yu = np.array([0,0,1,0, 1,1,1,0, 0,0,0,1, 1,1,0,1])

full = DecisionTreeClassifier(criterion="gini", random_state=1).fit(Xu, yu)
dump("u3-full", full)

for msl in (2, 3):
    t = DecisionTreeClassifier(criterion="gini", random_state=1,
                               min_samples_leaf=msl).fit(Xu, yu)
    print(f"u3-msl{msl}: depth={t.get_depth()} leaves={t.get_n_leaves()}")
t = DecisionTreeClassifier(criterion="gini", random_state=1,
                           min_samples_split=6).fit(Xu, yu)
print(f"u3-mss6: depth={t.get_depth()} leaves={t.get_n_leaves()}")

for mln in (3, 5):
    t = DecisionTreeClassifier(criterion="gini", random_state=1,
                               max_leaf_nodes=mln).fit(Xu, yu)
    dump(f"u3-mln{mln}", t)

t = DecisionTreeClassifier(criterion="gini", random_state=1,
                           min_impurity_decrease=0.05).fit(Xu, yu)
dump("u3-mid.05", t)

# ccp path + pruned trees along it.
path = full.cost_complexity_pruning_path(Xu, yu)
print("u3-ccp alphas:", [repr(a) for a in path.ccp_alphas])
print("u3-ccp impurities:", [repr(a) for a in path.impurities])
for a in path.ccp_alphas[1:-1]:
    t = DecisionTreeClassifier(criterion="gini", random_state=1,
                               ccp_alpha=a + 1e-12).fit(Xu, yu)
    print(f"u3-ccp@{a!r}: depth={t.get_depth()} leaves={t.get_n_leaves()}")

# class_weight: weights that FLIP the chosen split (no pure split exists,
# so weighting the minority class moves the best boundary): unweighted root
# 4.5 -> weighted {0:1, 1:4} root 1.5, and leaf fractions are WEIGHTED
# (right = 3x0 + 3x1 -> 3/15, 12/15).
Xw = np.arange(8.0).reshape(-1, 1)
yw = np.array([0, 0, 1, 0, 0, 1, 1, 0])
uw = DecisionTreeClassifier(criterion="gini", random_state=1, max_depth=1).fit(Xw, yw)
dump("u3-cw-none-d1", uw)
cw = DecisionTreeClassifier(criterion="gini", random_state=1, max_depth=1,
                            class_weight={0: 1.0, 1: 4.0}).fit(Xw, yw)
dump("u3-cw-1:4-d1", cw)
print("  cw proba:", [[repr(v) for v in row] for row in cw.predict_proba(np.array([[0.0], [5.0]]))])
