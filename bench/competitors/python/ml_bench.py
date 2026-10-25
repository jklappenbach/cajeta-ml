#!/usr/bin/env python3
"""The competitor side of the dev.cajeta.ml benchmarks: scikit-learn for the
classical page, PyTorch for the neural one, emitting the same ids, variants
and schema-v2 columns as the Cajeta harness.

    ml_bench.py --data <dir> --out <python.csv> [--bench <name>] [--list]

Reads only what gen_data.py wrote into <dir>. Each row's check is the same
check the Cajeta row makes, against the same reference file, with the same
tolerance (TOL below, mirrored in the Cajeta sources' Tol class).

CAJETA_BENCH_NO_GPU=1: every `device` row is skipped and torch.cuda is never
called — not even is_available(), which initializes the driver.
"""
import argparse
import os
import platform
import sys
import time

import numpy as np

import benchcsv as bc

# Tolerances: identical to bench/src/main/cajeta/bench/ml/Tol.cajeta.
TOL = {
    "linear-regression": 1e-9,      # |R2 - R2_ref|
    "logistic-regression": 0.005,   # |accuracy - accuracy_ref|
    "kmeans-synthetic": 1e-6,       # |inertia/inertia_ref - 1|
    "kmeans-digits": 0.05,          # |inertia/inertia_ref - 1|
    "knn": 0.002,                   # |accuracy - accuracy_ref|
    "pca": 1e-8,                    # max |ratio - ratio_ref| / ratio_ref
    "standard-scaler": 1e-10,       # max |z - z_ref| over the first 1000 rows
    "mlp-logits": 1e-4,             # max |logit - logit_ref|
    "cnn-logits": 1e-4,
    "transformer-out": 1e-3,
    "grad": 1e-4,                   # max |dW - dW_ref|
}

# warmup, trials: the Cajeta side's warmupIters()/measuredIters(), row for row.
WT = {
    "linear-regression": (3, 10), "logistic-regression": (3, 10), "kmeans": (3, 10),
    "knn": (2, 5), "pca": (3, 10), "standard-scaler": (3, 10),
    "mlp-train-step": (5, 50), "mlp-infer": (5, 50), "cnn-train-step": (3, 20),
    "cnn-infer": (3, 20), "transformer-block-train": (2, 5), "transformer-block-infer": (2, 5),
    "autodiff-overhead": (5, 50), "checkpoint-load": (2, 10),
}

CLASSICAL = ["linear-regression", "logistic-regression", "kmeans", "knn", "pca", "standard-scaler"]
NEURAL = ["mlp-train-step", "mlp-infer", "cnn-train-step", "cnn-infer",
          "transformer-block-train", "transformer-block-infer", "autodiff-overhead",
          "checkpoint-load"]
VARIANTS = {
    "linear-regression": ["synthetic", "housing"],
    "logistic-regression": ["synthetic", "digits"],
    "kmeans": ["synthetic", "digits"],
    "knn": ["synthetic"], "pca": ["synthetic"], "standard-scaler": ["synthetic"],
    "autodiff-overhead": ["cpu"], "checkpoint-load": ["cpu"],
}
for _n in NEURAL:
    VARIANTS.setdefault(_n, ["cpu", "device"])


def load(d, rel):
    return np.load(os.path.join(d, rel))


def have(d, rel):
    return os.path.exists(os.path.join(d, rel))


def size_note(x):
    return "x".join(str(v) for v in x.shape)


# ---------------------------------------------------------------- classical

def sk_meta():
    import sklearn
    return dict(library="scikit-learn", library_version=sklearn.__version__,
                platform=f"scikit-learn {sklearn.__version__}")


def run_classical(name, variant, d):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    base = f"{name}/{variant}"
    meta = dict(benchmark=name, area="ml", variant=variant, **sk_meta())
    w, t = bc.warmup_trials(*WT[name])
    if not have(d, base):
        return bc.skipped_row(f"dataset {base} not generated (gen_data.py)", **meta)
    ref = load(d, f"ref/{name}-{variant}.npy")
    state = {}

    if name in ("linear-regression", "logistic-regression", "knn"):
        xtr, ytr = load(d, f"{base}/x_train.npy"), load(d, f"{base}/y_train.npy")
        xte, yte = load(d, f"{base}/x_test.npy"), load(d, f"{base}/y_test.npy")
        ds = f"{name}-{variant} train {size_note(xtr)} test {size_note(xte)}"

        def make():
            if name == "linear-regression":
                return LinearRegression()
            if name == "logistic-regression":
                return LogisticRegression(C=1.0, solver="newton-cholesky", tol=1e-4, max_iter=100)
            return KNeighborsClassifier(5, algorithm="brute")

        def body():
            m = make().fit(xtr, ytr)
            state["pred"] = m.predict(xte)

        samples = bc.measure(body, w, t)
        if name == "linear-regression":
            from sklearn.metrics import r2_score
            got = r2_score(yte, state["pred"])
            tol = TOL[name]
            what = "R2"
        else:
            got = float(np.mean(state["pred"] == yte))
            tol = TOL[name]
            what = "accuracy"
        diff = abs(got - ref[0])
        ok = diff <= tol
        note = f"{what} {got:.10g} vs scikit-learn {ref[0]:.10g}, |d| {diff:.3g} <= {tol:g}"
        return bc.measured_row(samples, w, input_size=xtr.shape[0], unit="rows/s", check_ok=ok, note=note,
                               dataset_name=ds, **meta)

    if name == "kmeans":
        x = load(d, f"{base}/x.npy")
        ds = f"kmeans-{variant} {size_note(x)}"

        def body():
            state["m"] = KMeans(10, n_init=1, random_state=20261025).fit(x)

        samples = bc.measure(body, w, t)
        got = state["m"].inertia_
        rel = abs(got / ref[0] - 1.0)
        tol = TOL[f"kmeans-{variant}"]
        note = (f"k=10 to convergence ({state['m'].n_iter_} iterations); inertia {got:.10g} vs "
                f"reference {ref[0]:.10g}, rel |d| {rel:.3g} <= {tol:g}")
        return bc.measured_row(samples, w, input_size=x.shape[0], unit="rows/s", check_ok=rel <= tol, note=note,
                               dataset_name=ds, **meta)

    if name == "pca":
        x = load(d, f"{base}/x.npy")
        k = int(load(d, f"{base}/k.npy")[0])

        def body():
            state["m"] = PCA(k, svd_solver="full").fit(x)

        samples = bc.measure(body, w, t)
        got = state["m"].explained_variance_ratio_
        rel = float(np.max(np.abs(got - ref) / ref))
        tol = TOL[name]
        note = f"{k} components; explained-variance ratio max rel |d| {rel:.3g} <= {tol:g}"
        return bc.measured_row(samples, w, input_size=x.shape[0], unit="rows/s", check_ok=rel <= tol, note=note,
                               dataset_name=f"pca-{variant} {size_note(x)}", **meta)

    if name == "standard-scaler":
        x = load(d, f"{base}/x.npy")

        def body():
            state["z"] = StandardScaler().fit(x).transform(x)

        samples = bc.measure(body, w, t)
        diff = float(np.max(np.abs(state["z"][:1000] - ref)))
        tol = TOL[name]
        note = f"fit and transform; max |z - z_ref| {diff:.3g} <= {tol:g} (first 1000 rows)"
        return bc.measured_row(samples, w, input_size=x.shape[0], unit="rows/s",
                               check_ok=diff <= tol, note=note,
                               dataset_name=f"standard-scaler-{variant} {size_note(x)}", **meta)
    raise ValueError(name)


# ---------------------------------------------------------------- neural

def torch_meta(device):
    import torch
    m = dict(library="torch", library_version=torch.__version__,
             platform=f"torch {torch.__version__} {'cuda' if device else 'cpu'}")
    return m


def torch_threads_note():
    import torch
    return f"torch intra-op threads {torch.get_num_threads()}"


def run_neural(name, variant, d):
    device = variant == "device"
    meta = dict(benchmark=name, area="ml", variant=variant)
    if device and bc.no_gpu():
        # Never import a CUDA path: the skip is decided before torch is asked anything.
        import torch
        return bc.skipped_row(bc.GPU_HELD, backend="gpu", library="torch",
                              library_version=torch.__version__,
                              platform=f"torch {torch.__version__} cuda", **meta)
    import torch
    import torch.nn.functional as F
    import twins

    dev = torch.device("cpu")
    if device:
        if not torch.cuda.is_available():
            return bc.skipped_row("no CUDA device visible to torch", backend="cuda", **meta,
                                  **torch_meta(True))
        dev = torch.device("cuda")
    meta.update(torch_meta(device))
    meta["backend"] = "cuda" if device else "cpu"
    if device:
        meta["driver"] = f"CUDA {torch.version.cuda}"

    def sync():
        if device:
            torch.cuda.synchronize()

    steps, loads = (int(v) for v in load(d, "config/train.npy"))
    w, t = bc.warmup_trials(*WT[name])
    torch.manual_seed(20261025)

    def model_and_batch(kind):
        if kind == "mlp":
            n_in, hid, n_out = (int(v) for v in load(d, "mlp/config.npy"))
            m = twins.Mlp(n_in, [hid], n_out)
            mdl = f"mlp {n_in}-{hid}-{n_out} f32"
        elif kind == "cnn":
            c, wd, cls = (int(v) for v in load(d, "cnn/config.npy"))
            m = twins.SmallCnn(c, wd, cls)
            mdl = f"small-cnn c{c} w{wd} k{cls} f32"
        else:
            e, h, ff = (int(v) for v in load(d, "transformer/config.npy"))
            m = twins.EncoderBlock(e, h, ff)
            mdl = f"encoder-block d{e} h{h} ff{ff} f32"
        m.load_state_dict(torch.load(os.path.join(d, f"{kind}/model.pt"), weights_only=True),
                          strict=False)
        m = m.to(dev)
        x = torch.from_numpy(load(d, f"{kind}/x.npy")).to(dev)
        y = torch.from_numpy(load(d, f"{kind}/y.npy")).to(dev)
        return m, x, y, mdl

    kind = "transformer" if name.startswith("transformer") else name.split("-")[0]
    if name in ("mlp-train-step", "cnn-train-step", "transformer-block-train"):
        m, x, y, mdl = model_and_batch(kind)
        m.train()
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        losses = []

        def step():
            opt.zero_grad(set_to_none=True)
            out = m(x)
            loss = F.mse_loss(out, y) if kind == "transformer" else F.cross_entropy(out, y.long())
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
            sync()

        samples = bc.measure(step, w, t)
        while len(losses) < steps:
            step()
        ok = losses[steps - 1] < losses[0]
        note = (f"Adam lr 1e-3; loss {losses[0]:.6g} -> {losses[steps - 1]:.6g} over {steps} steps; "
                + torch_threads_note())
        return bc.measured_row(samples, w, input_size=x.shape[0], unit="samples/s", check_ok=ok,
                               note=note, model=mdl, dataset_name=f"{kind} batch {size_note(x)}",
                               **meta)

    if name in ("mlp-infer", "cnn-infer", "transformer-block-infer"):
        m, x, y, mdl = model_and_batch(kind)
        m.eval()
        state = {}

        def infer():
            with torch.no_grad():
                state["out"] = m(x)
            sync()

        samples = bc.measure(infer, w, t)
        refn = {"mlp": "mlp-logits", "cnn": "cnn-logits", "transformer": "transformer-out"}[kind]
        ref = load(d, f"ref/{refn}.npy")
        diff = float(np.max(np.abs(state["out"].cpu().numpy() - ref)))
        tol = TOL[refn]
        note = f"eval; max |out - ref| {diff:.3g} <= {tol:g}; " + torch_threads_note()
        return bc.measured_row(samples, w, input_size=x.shape[0], unit="samples/s",
                               check_ok=diff <= tol, note=note, model=mdl,
                               dataset_name=f"{kind} batch {size_note(x)}", **meta)

    if name == "autodiff-overhead":
        m, x, y, mdl = model_and_batch("mlp")
        fwd, bwd = [], []

        def fb():
            m.zero_grad(set_to_none=True)
            t0 = time.perf_counter_ns()
            loss = F.cross_entropy(m(x), y.long())
            t1 = time.perf_counter_ns()
            loss.backward()
            t2 = time.perf_counter_ns()
            fwd.append(t1 - t0)
            bwd.append(t2 - t1)

        samples = bc.measure(fb, w, t)
        f_med, b_med = sorted(fwd)[len(fwd) // 2], sorted(bwd)[len(bwd) // 2]
        worst = 0.0
        for k, p in m.named_parameters():
            worst = max(worst, float(np.max(np.abs(twins.cajeta_layout(k, p.grad)
                                                   - load(d, f"ref/mlp-grad/{k}.npy")))))
        tol = TOL["grad"]
        note = (f"forward+backward; backward/forward {b_med / max(f_med, 1):.3f} "
                f"(forward median {f_med} ns, backward median {b_med} ns); "
                f"max |dW - dW_ref| {worst:.3g} <= {tol:g}; " + torch_threads_note())
        return bc.measured_row(samples, w, input_size=x.shape[0], unit="samples/s", check_ok=worst <= tol,
                               note=note, model=mdl, **meta)

    if name == "checkpoint-load":
        n_in, hid, n_out = (int(v) for v in load(d, "mlp/config.npy"))
        path = os.path.join(d, "mlp/model.pt")
        m = twins.Mlp(n_in, [hid], n_out)

        def body():
            for _ in range(loads):
                m.load_state_dict(torch.load(path, weights_only=True))

        samples = bc.measure(body, w, t)
        equal = all(np.array_equal(twins.cajeta_layout(k, p), load(d, f"mlp/cajeta/{k}.npy"))
                    for k, p in m.named_parameters())
        note = f"torch.load + load_state_dict x{loads} per iteration; weights equal: {equal}"
        return bc.measured_row(samples, w, input_size=loads, unit="loads/s", check_ok=equal, note=note,
                               model=f"mlp {n_in}-{hid}-{n_out} f32",
                               dataset_name=f"mlp/model.pt {os.path.getsize(path)} B", **meta)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False)
    ap.add_argument("--out", default="python.csv")
    ap.add_argument("--bench")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    names = [n for n in CLASSICAL + NEURAL if a.bench in (None, n)]
    if a.list:
        for n in names:
            print(f"  ml/{n}  {VARIANTS[n]}")
        return 0
    if not a.data:
        ap.error("--data is required with a run")
    rows, bad = [], 0
    for n in names:
        for v in VARIANTS[n]:
            try:
                r = run_classical(n, v, a.data) if n in CLASSICAL else run_neural(n, v, a.data)
            except Exception as e:  # a failing benchmark never stops the run
                r = bc.failed_row(f"{type(e).__name__}: {e}", benchmark=n, area="ml", variant=v)
            print(f"  {r['status']}  ml/{n} [python {r['library']}] variant='{v}'  "
                  f"median_ns={r['median_ns']}  {r['notes']}", flush=True)
            bad += r["status"] in ("failed", "invalid")
            rows.append(r)
    bc.write(a.out, rows)
    print(f"ml_bench.py: wrote {a.out} ({len(rows)} rows)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
