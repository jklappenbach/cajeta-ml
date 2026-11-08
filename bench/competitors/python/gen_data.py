#!/usr/bin/env python3
"""Writes every input the dev.cajeta.ml benchmarks read, once, from one seed,
plus scikit-learn's and torch's reference results on that same data.

    gen_data.py <out-dir>        (bench.sh: data/smoke or data/full)

Both sides read these files and nothing else: the Cajeta benchmarks through
cajeta.math.npio.Npy and dev.cajeta.ml.io.PtReader, the Python runner through
numpy and torch.load. The reference results are what the Cajeta rows' checks
compare against (spec §7.2: scikit-learn's metric is the reference for the
classical rows; torch's logits and gradients for the neural ones, and its
forward of each op for the per-op rows).

Sizes follow the pages (bench/ml/Classical.md, Neural.md) unless
CAJETA_BENCH_SMOKE=1, which shrinks them (recorded in manifest.tsv and in each
row's notes). The seed is SEED below; every generator is seeded from it.

The GPU is never touched here: torch runs on the CPU only.
"""
import hashlib
import os
import sys

import numpy as np

import benchcsv

SEED = 20261025


def sizes(smoke):
    if smoke:
        return dict(n=10_000, p=20, knn_train=10_000, knn_q=1_000, pca_k=10,
                    mlp=(784, 512, 10, 64), cnn=(1, 8, 10, 28, 28, 64),
                    tfm=(64, 4, 256, 16, 8), steps=20, loads=10, digits_train=360)
    return dict(n=100_000, p=100, knn_train=100_000, knn_q=10_000, pca_k=20,
                mlp=(784, 512, 10, 64), cnn=(1, 16, 10, 28, 28, 64),
                tfm=(512, 8, 2048, 128, 64), steps=100, loads=100, digits_train=0)


class Out:
    def __init__(self, root):
        self.root = root
        self.lines = []

    def save(self, rel, arr, dtype):
        path = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        a = np.ascontiguousarray(np.asarray(arr, dtype=dtype))
        np.save(path, a)
        self.lines.append((rel, "x".join(map(str, a.shape)) or "scalar", self.sha(path)))

    def f64(self, rel, arr):
        self.save(rel, arr, np.float64)

    def f32(self, rel, arr):
        self.save(rel, arr, np.float32)

    def i64(self, rel, arr):
        self.save(rel, arr, np.int64)

    def file(self, rel):
        self.lines.append((rel, "file", self.sha(os.path.join(self.root, rel))))

    @staticmethod
    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()


def split(rng, x, y, test=0.2):
    idx = rng.permutation(x.shape[0])
    nt = int(np.ceil(test * x.shape[0]))
    te, tr = idx[:nt], idx[nt:]
    return x[tr], y[tr], x[te], y[te]


def classical(o, s, rng):
    from sklearn.cluster import KMeans
    from sklearn.datasets import fetch_california_housing, load_digits, make_blobs
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    n, p = s["n"], s["p"]

    # linear-regression / synthetic: y = Xw + 3 + noise
    x = rng.standard_normal((n, p))
    w = rng.standard_normal(p)
    y = x @ w + 3.0 + 0.5 * rng.standard_normal(n)
    xtr, ytr, xte, yte = split(rng, x, y)
    for k, v in dict(x_train=xtr, y_train=ytr, x_test=xte, y_test=yte).items():
        o.f64(f"linear-regression/synthetic/{k}.npy", v)
    r2 = LinearRegression().fit(xtr, ytr).score(xte, yte)
    o.f64("ref/linear-regression-synthetic.npy", [r2])

    # linear-regression / housing: the California housing set (sklearn cache)
    try:
        h = fetch_california_housing()
        xtr, ytr, xte, yte = split(rng, h.data, h.target)
        for k, v in dict(x_train=xtr, y_train=ytr, x_test=xte, y_test=yte).items():
            o.f64(f"linear-regression/housing/{k}.npy", v)
        r2 = LinearRegression().fit(xtr, ytr).score(xte, yte)
        o.f64("ref/linear-regression-housing.npy", [r2])
    except Exception as e:  # no network and no cache: both sides skip it
        print(f"gen_data: California housing unavailable ({e}); the housing variant will skip",
              file=sys.stderr)

    # logistic-regression / synthetic: binary labels from a noisy linear score
    x = rng.standard_normal((n, p))
    w = rng.standard_normal(p)
    y = ((x @ w + 1.0 * rng.standard_normal(n)) > 0).astype(np.float64)
    xtr, ytr, xte, yte = split(rng, x, y)
    for k, v in dict(x_train=xtr, y_train=ytr, x_test=xte, y_test=yte).items():
        o.f64(f"logistic-regression/synthetic/{k}.npy", v)
    acc = LogisticRegression(C=1.0, solver="newton-cholesky", tol=1e-4, max_iter=100) \
        .fit(xtr, ytr).score(xte, yte)
    o.f64("ref/logistic-regression-synthetic.npy", [acc])

    # logistic-regression / digits: 10 classes, one multinomial likelihood
    d = load_digits()
    xtr, ytr, xte, yte = split(rng, d.data.astype(np.float64), d.target.astype(np.float64))
    if s["digits_train"]:
        # Smoke only: the multinomial Newton fit is O(n K^2 p^2) per iteration
        # and took 72 s per fit on the full 1437-row split (Phoenix, cajeta
        # 0.46.3), so the smoke trains on the first digits_train rows.
        xtr, ytr = xtr[:s["digits_train"]], ytr[:s["digits_train"]]
    for k, v in dict(x_train=xtr, y_train=ytr, x_test=xte, y_test=yte).items():
        o.f64(f"logistic-regression/digits/{k}.npy", v)
    acc = LogisticRegression(C=1.0, solver="newton-cholesky", tol=1e-4, max_iter=100) \
        .fit(xtr, ytr).score(xte, yte)
    o.f64("ref/logistic-regression-digits.npy", [acc])

    # kmeans / synthetic: 10 separated blobs; kmeans / digits: the digits pixels
    xb, _ = make_blobs(n_samples=n, n_features=p, centers=10, cluster_std=1.0,
                       center_box=(-20, 20), random_state=SEED)
    o.f64("kmeans/synthetic/x.npy", xb)
    o.f64("ref/kmeans-synthetic.npy",
          [KMeans(10, n_init=1, random_state=SEED).fit(xb).inertia_])
    xd = d.data.astype(np.float64)
    o.f64("kmeans/digits/x.npy", xd)
    o.f64("ref/kmeans-digits.npy",
          [KMeans(10, n_init=1, random_state=SEED).fit(xd).inertia_])

    # knn / synthetic: k = 5 over knn_q queries against knn_train labelled rows,
    # 10 overlapping blobs so accuracy is neither 0 nor 1
    m = s["knn_train"] + s["knn_q"]
    xk, yk = make_blobs(n_samples=m, n_features=p, centers=10, cluster_std=4.0,
                        center_box=(-10, 10), random_state=SEED + 1)
    perm = rng.permutation(m)
    xk, yk = xk[perm], yk[perm].astype(np.float64)
    q = s["knn_q"]
    o.f64("knn/synthetic/x_train.npy", xk[q:])
    o.f64("knn/synthetic/y_train.npy", yk[q:])
    o.f64("knn/synthetic/x_test.npy", xk[:q])
    o.f64("knn/synthetic/y_test.npy", yk[:q])
    acc = KNeighborsClassifier(5, algorithm="brute").fit(xk[q:], yk[q:]).score(xk[:q], yk[:q])
    o.f64("ref/knn-synthetic.npy", [acc])

    # pca / synthetic: a decaying spectrum (distinct eigenvalues) plus noise
    scale = np.geomspace(10.0, 0.1, p)
    xp = (rng.standard_normal((n, p)) * scale) @ np.linalg.qr(rng.standard_normal((p, p)))[0]
    o.f64("pca/synthetic/x.npy", xp)
    o.i64("pca/synthetic/k.npy", [s["pca_k"]])
    o.f64("ref/pca-synthetic.npy", PCA(s["pca_k"], svd_solver="full").fit(xp).explained_variance_ratio_)

    # standard-scaler / synthetic: columns with their own mean and scale
    xs = rng.standard_normal((n, p)) * rng.uniform(0.1, 100.0, p) + rng.uniform(-50, 50, p)
    o.f64("standard-scaler/synthetic/x.npy", xs)
    o.f64("ref/standard-scaler-synthetic.npy", StandardScaler().fit(xs).transform(xs)[:1000])


def neural(o, s, rng):
    import torch
    import torch.nn.functional as F
    import twins

    torch.manual_seed(SEED)
    torch.set_grad_enabled(True)
    n_in, hid, n_out, b = s["mlp"]
    o.i64("config/train.npy", [s["steps"], s["loads"]])

    # MLP 784-512-10, batch 64: weights, batch, eval logits, CE gradients
    mlp = twins.Mlp(n_in, [hid], n_out)
    x = rng.standard_normal((b, n_in)).astype(np.float32)
    y = rng.integers(0, n_out, b).astype(np.float32)
    o.i64("mlp/config.npy", [n_in, hid, n_out])
    o.f32("mlp/x.npy", x)
    o.f32("mlp/y.npy", y)
    sd = twins.params_only(mlp)
    torch.save(sd, os.path.join(o.root, "mlp/model.pt"))
    o.file("mlp/model.pt")
    for k, v in sd.items():
        o.f32(f"mlp/cajeta/{k}.npy", twins.cajeta_layout(k, v))
    with torch.no_grad():
        o.f32("ref/mlp-logits.npy", mlp(torch.from_numpy(x)).numpy())
    mlp.zero_grad()
    F.cross_entropy(mlp(torch.from_numpy(x)), torch.from_numpy(y).long()).backward()
    for k, v in mlp.named_parameters():
        o.f32(f"ref/mlp-grad/{k}.npy", twins.cajeta_layout(k, v.grad))

    # SmallCnn on 28x28, batch 64
    c, wd, cls, hh, ww, bc = s["cnn"]
    cnn = twins.SmallCnn(c, wd, cls)
    xc = rng.standard_normal((bc, c, hh, ww)).astype(np.float32)
    yc = rng.integers(0, cls, bc).astype(np.float32)
    o.i64("cnn/config.npy", [c, wd, cls])
    o.f32("cnn/x.npy", xc)
    o.f32("cnn/y.npy", yc)
    torch.save(twins.params_only(cnn), os.path.join(o.root, "cnn/model.pt"))
    o.file("cnn/model.pt")
    cnn.eval()
    with torch.no_grad():
        o.f32("ref/cnn-logits.npy", cnn(torch.from_numpy(xc)).numpy())

    # one pre-norm encoder block, d = embed, seq, batch
    e, heads, ff, t, bt = s["tfm"]
    blk = twins.EncoderBlock(e, heads, ff)
    xt = rng.standard_normal((bt, t, e)).astype(np.float32)
    yt = rng.standard_normal((bt, t, e)).astype(np.float32)
    o.i64("transformer/config.npy", [e, heads, ff])
    o.f32("transformer/x.npy", xt)
    o.f32("transformer/y.npy", yt)
    torch.save(twins.params_only(blk), os.path.join(o.root, "transformer/model.pt"))
    o.file("transformer/model.pt")
    with torch.no_grad():
        o.f32("ref/transformer-out.npy", blk(torch.from_numpy(xt)).numpy())


def ops(o, s, rng):
    """The per-op rows' inputs and torch's forward on each (ml/op-<name>): the
    transformer-shaped activation, its rows against a d x ff weight, the
    CNN-shaped activation with a 3x3 weight, and rows x d logits with targets.
    Weights are scaled so every output is O(1) and an absolute tolerance means
    the same thing on every row."""
    import torch
    import torch.nn.functional as F

    e, _heads, ff, t, bt = s["tfm"]
    c, wd, _cls, hh, ww, bc = s["cnn"]
    x = rng.standard_normal((bt, t, e)).astype(np.float32)
    y = rng.standard_normal((bt, t, e)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, e).astype(np.float32)
    beta = (0.1 * rng.standard_normal(e)).astype(np.float32)
    x2 = x.reshape(bt * t, e)
    w = (rng.standard_normal((e, ff)) / np.sqrt(e)).astype(np.float32)
    cx = rng.standard_normal((bc, wd, hh, ww)).astype(np.float32)
    cw = (rng.standard_normal((wd, wd, 3, 3)) / np.sqrt(9 * wd)).astype(np.float32)
    cb = (0.1 * rng.standard_normal(wd)).astype(np.float32)
    logits = (2.0 * rng.standard_normal((bt * t, e))).astype(np.float32)
    targets = rng.integers(0, e, bt * t).astype(np.float32)
    for k, v in dict(x=x, y=y, gamma=gamma, beta=beta, x2=x2, w=w, conv_x=cx, conv_w=cw,
                     conv_b=cb, logits=logits, targets=targets).items():
        o.f32(f"ops/{k}.npy", v)

    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    with torch.no_grad():
        refs = {
            "add": tx + ty,
            "relu": F.relu(tx),
            "gelu": F.gelu(tx),
            "tanh": torch.tanh(tx),
            "sigmoid": torch.sigmoid(tx),
            "softmax": F.softmax(tx, dim=-1),
            "layernorm": F.layer_norm(tx, (e,), torch.from_numpy(gamma), torch.from_numpy(beta), 1e-5),
            "matmul": torch.from_numpy(x2) @ torch.from_numpy(w),
            "conv2d": F.conv2d(torch.from_numpy(cx), torch.from_numpy(cw), torch.from_numpy(cb),
                               stride=1, padding=1),
            "maxpool2d": F.max_pool2d(torch.from_numpy(cx), 2),
            "cross-entropy": F.cross_entropy(torch.from_numpy(logits),
                                             torch.from_numpy(targets).long()).reshape(1),
        }
    for k, v in refs.items():
        o.f32(f"ref/op-{k}.npy", v.numpy())


def main(out):
    smoke = benchcsv.smoke()
    s = sizes(smoke)
    os.makedirs(out, exist_ok=True)
    o = Out(out)
    rng = np.random.default_rng(SEED)
    classical(o, s, rng)
    neural(o, s, rng)
    ops(o, s, rng)
    with open(os.path.join(out, "manifest.tsv"), "w") as f:
        f.write(f"# dev.cajeta.ml bench data, seed {SEED}, scale {'smoke' if smoke else 'full'}\n")
        f.write(f"# sizes {s}\n")
        f.write("file\tshape\tsha256\n")
        for rel, shape, sha in o.lines:
            f.write(f"{rel}\t{shape}\t{sha}\n")
    print(f"gen_data: wrote {len(o.lines)} files to {out} (seed {SEED})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: gen_data.py <out-dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
