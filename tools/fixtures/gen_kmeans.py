#!/usr/bin/env python3
# cajeta-ml v2 Unit 4 — KMeans fixture against PINNED sklearn 1.9.0.
# Three well-separated 2-D blobs, deterministic offsets: Lloyd converges to
# the exact blob partition from any sane init, so the pinned quantities are
# init-independent (inertia, centers=blob means, the partition itself).
import numpy as np
import sklearn
from sklearn.cluster import KMeans

print("sklearn", sklearn.__version__)

pts = []
centers = [(0.0, 0.0), (10.0, 10.0), (-10.0, 10.0)]
for c, (cx, cy) in enumerate(centers):
    for j in range(8):
        pts.append((cx + (j % 4) * 0.1, cy + (j % 3) * 0.1))
X = np.array(pts)

m = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(X)
print("inertia", repr(m.inertia_))
cs = sorted(map(tuple, m.cluster_centers_.round(12)))
for c in cs:
    print("center", c)
print("labels", m.labels_.tolist())
