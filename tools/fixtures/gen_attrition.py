#!/usr/bin/env python3
# ml-trees-ensembles U8 — the HR Employee Attrition reference pipeline
# (research/ml/course-material/practical-data-science/lvc1-decision-trees,
# Case_Study_Employee_Attrition_Prediction.ipynb), REGENERATED under pinned
# sklearn 1.9.0 (spec §10.2/§8.2.1: the notebook's committed outputs are from
# an older sklearn; version drift is the FIRST suspect on any disagreement,
# so only regenerated numbers are pinned — the notebook stays out of git).
#
# Seeding audit (§10.3): the pinned pipeline is fully seeded —
# train_test_split(random_state=1, stratify=Y) and
# DecisionTreeClassifier(random_state=1). The notebook's GridSearchCV-tuned
# variants and the RF/AdaBoost/GB/XGB cells are NOT pinned here (their
# sklearn-internal RNG streams are a recorded non-goal; see plan 6.1.3).
#
# Run: /home/julian/code/ml/venv-sklearn-ref/bin/python gen_attrition.py
# Writes the exact train/test matrices to build/attrition/ (GITIGNORED — the
# cajeta test loads them via ML_ATTRITION_DIR and self-skips without).
import os
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score

print("sklearn", sklearn.__version__)

NB = ("/home/julian/code/cpp/cajeta-six/research/ml/course-material/"
      "practical-data-science/lvc1-decision-trees/datasets/"
      "HR_Employee_Attrition_Dataset.xlsx")
df = pd.read_excel(NB)
df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
dummies = ['BusinessTravel', 'Department', 'EducationField', 'Gender',
           'MaritalStatus', 'JobRole']
df = pd.get_dummies(data=df, columns=dummies, drop_first=True)
df['OverTime'] = df.OverTime.map({'Yes': 1, 'No': 0})
df['Attrition'] = df.Attrition.map({'Yes': 1, 'No': 0})
Y = df.Attrition
X = df.drop(['Attrition'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1, stratify=Y)

dt = DecisionTreeClassifier(class_weight={0: 0.17, 1: 0.83}, random_state=1)
dt.fit(x_train, y_train)

for name, xs, ys in (("train", x_train, y_train), ("test", x_test, y_test)):
    p = dt.predict(xs)
    print(f"u8-dt {name}: acc={accuracy_score(ys, p)!r} "
          f"recall1={recall_score(ys, p)!r} prec1={precision_score(ys, p)!r}")
print("u8-dt depth:", dt.get_depth(), "leaves:", dt.get_n_leaves())
print("u8 shapes:", x_train.shape, x_test.shape)

out = os.path.join(os.path.dirname(__file__), "..", "..", "build", "attrition")
os.makedirs(out, exist_ok=True)
# ascontiguousarray: pandas' to_numpy on a mixed-block frame yields a
# FORTRAN-ORDER array and np.save records fortran_order=True — which
# cajeta.math.npio.Npy silently misreads as C-order (INDEX row
# npy-fortran-order-silent-misread). Always save C-order.
np.save(os.path.join(out, "x_train.npy"), np.ascontiguousarray(x_train.to_numpy(dtype=np.float64)))
np.save(os.path.join(out, "y_train.npy"), np.ascontiguousarray(y_train.to_numpy(dtype=np.float64)))
np.save(os.path.join(out, "x_test.npy"), np.ascontiguousarray(x_test.to_numpy(dtype=np.float64)))
np.save(os.path.join(out, "y_test.npy"), np.ascontiguousarray(y_test.to_numpy(dtype=np.float64)))
print("wrote", out)
