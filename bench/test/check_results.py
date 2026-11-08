#!/usr/bin/env python3
"""Asserts a dev.cajeta.ml bench run against the pages (bench_smoke.sh).

    check_results.py <run-dir> <--list output>

The table below is the contract of cajeta bench/ml/Classical.md and Neural.md:
the id, its variants, and the competitor platform that must stand beside it.
"""
import csv
import os
import sys

HEADER = (
    "schema_version,run_id,timestamp_unix,benchmark,area,dataset_name,dataset_bytes,"
    "dataset_sha256,input_size,language,language_version,library,library_version,flags,"
    "warmup,trials,min_ns,median_ns,mean_ns,p95_ns,throughput,throughput_unit,peak_rss_kb,"
    "cgroup_peak_kb,alloc_bytes,alloc_count,working_set_kb,status,check_ok,notes,variant,"
    "backend,machine,os,driver,model,platform"
)

# id -> (variants, competitor library)
EXPECTED = {
    "linear-regression":        (["synthetic", "housing"], "scikit-learn"),
    "logistic-regression":      (["synthetic", "digits"], "scikit-learn"),
    "kmeans":                   (["synthetic", "digits"], "scikit-learn"),
    "knn":                      (["synthetic"], "scikit-learn"),
    "pca":                      (["synthetic"], "scikit-learn"),
    "standard-scaler":          (["synthetic"], "scikit-learn"),
    "mlp-train-step":           (["cpu", "device"], "torch"),
    "mlp-infer":                (["cpu", "device"], "torch"),
    "cnn-train-step":           (["cpu", "device"], "torch"),
    "cnn-infer":                (["cpu", "device"], "torch"),
    "transformer-block-train":  (["cpu", "device"], "torch"),
    "transformer-block-infer":  (["cpu", "device"], "torch"),
    "autodiff-overhead":        (["cpu"], "torch"),
    "checkpoint-load":          (["cpu"], "torch"),
}
for _op in ("add", "relu", "gelu", "tanh", "sigmoid", "softmax", "layernorm", "matmul",
            "conv2d", "maxpool2d", "cross-entropy"):
    EXPECTED[f"op-{_op}"] = (["cpu", "device"], "torch")
GPU_HELD = "GPU held by another session (CAJETA_BENCH_NO_GPU=1)"
# A competitor whose runtime cannot run on this box's device is a recorded gap, not a
# failed smoke: the row says so (ml_bench.py probes torch in a child process first).
TORCH_BROKEN = "torch cannot run on this device"

failures = []


def check(ok, what):
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    if not ok:
        failures.append(what)


def main(run_dir, list_out):
    no_gpu = os.environ.get("CAJETA_BENCH_NO_GPU", "") in ("1", "true", "yes")
    listed = sorted(l.split()[0] for l in list_out.splitlines() if l.strip().startswith("ml/"))
    check(listed == sorted("ml/" + k for k in EXPECTED),
          f"--list names exactly the {len(EXPECTED)} page ids (got {listed})")

    for name in ("cajeta.csv", "python.csv", "results.csv"):
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            check(False, f"{name} exists")
            continue
        with open(path, newline="") as f:
            first = f.readline().rstrip("\n")
        check(first == HEADER, f"{name} carries the exact schema-v2 header")
    check(os.path.exists(os.path.join(run_dir, "env.csv")), "env.csv exists")

    with open(os.path.join(run_dir, "results.csv"), newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if "smoke" not in [p.strip() for p in r["notes"].split(";")]:
            check(False, f"row {r['area']}/{r['benchmark']} [{r['language']}] {r['variant']} is tagged smoke")

    for name, (variants, lib) in EXPECTED.items():
        for v in variants:
            for side in ("cajeta", lib):
                if side == "cajeta":
                    hit = [r for r in rows if r["language"] == "cajeta"
                           and r["benchmark"] == name and r["variant"] == v]
                else:
                    hit = [r for r in rows if r["language"] == "python" and r["library"] == lib
                           and r["benchmark"] == name and r["variant"] == v]
                label = f"ml/{name} [{side}] variant={v}"
                if len(hit) != 1:
                    check(False, f"{label}: exactly one row (found {len(hit)})")
                    continue
                r = hit[0]
                check(r["area"] == "ml", f"{label}: area is ml")
                if v == "device" and no_gpu:
                    check(r["status"] == "skipped" and GPU_HELD in r["notes"],
                          f"{label}: skipped while the GPU is held ({r['status']}: {r['notes']})")
                elif (v == "device" and side != "cajeta" and r["status"] == "skipped"
                      and TORCH_BROKEN in r["notes"]):
                    print(f"  [GAP]  {label}: {r['notes']}")
                else:
                    check(r["status"] == "ok" and r["check_ok"] == "true",
                          f"{label}: ok and checked ({r['status']} check_ok={r['check_ok']}: {r['notes']})")
                    check(r["median_ns"] not in ("", "-1"), f"{label}: has a median")

    print(f"\n{'OK' if not failures else 'FAILED'}: {len(failures)} failing check(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: check_results.py <run-dir> <--list output>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
