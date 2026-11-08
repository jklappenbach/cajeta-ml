# cajeta-ml/bench

`dev.cajeta.ml` measured against scikit-learn (the classical half) and PyTorch
(the neural half), on the shared harness `dev.cajeta.bench`. Every id, workload
and check is defined by a page in the compiler repo: `bench/ml/Classical.md`
and `bench/ml/Neural.md` (spec `specs/benchmark-reference-spec.md` §7.2).

## Run

```sh
cd bench
./bench.sh                               # Cajeta rows only
./bench.sh --competitors                 # and the scikit-learn / torch rows
./bench.sh --bench kmeans --competitors  # one id, both sides
CAJETA_BENCH_SMOKE=1 ./bench.sh --competitors   # reduced sizes, warmup 1, trials 3
./test/bench_smoke.sh                    # the smoke gate (asserts the pages' contract)
```

`bench.sh` does, in order, each step timed on stdout:

1. **data** — `competitors/python/gen_data.py` writes every input and every
   reference result once, from the recorded seed `20261025`, into
   `data/smoke/` or `data/full/` (gitignored). Both sides read these files and
   nothing else. `data-manifest.tsv` in the run directory lists every file with
   its shape and sha256.
2. **build** — `cajeta release` → `build/ml-bench` (`NO_BUILD=1` reuses it).
3. **cajeta** — `build/ml-bench --run` → `cajeta.csv` and `env.csv`.
4. **python** (`--competitors`) — `competitors/python/ml_bench.py` → `python.csv`.
5. **merge** — `results.csv` is both, under one schema-v2 header; `env.csv`
   gains the competitor versions, load average, GPU state, data seed and
   manifest checksum.

Output lands in `results/<machine>/<run-id>/` (gitignored: only the publish
step copies a non-smoke run into the compiler repo, spec §9.4) (`run-id` is
`<UTC basic>-<cajeta version>`). `cajeta bench` alone builds and measures the
Cajeta rows into `build/results.csv`; it needs the data step first.

Defaults: the compiler is `cajeta` on PATH, else `../../cajeta/build/src/cajeta`
(`CAJETA=` overrides); Python is the compiler repo's competitor venv
`../../cajeta/bench/competitors/python/.venv` (`BENCH_PYTHON=` overrides:
scikit-learn 1.9.1, torch 2.14.1+cu130, numpy).

| Environment | |
|---|---|
| `CAJETA_BENCH_SMOKE=1` | the smoke sizes below, warmup 1, trials 3, every row tagged `smoke` (never published) |
| `CAJETA_BENCH_NO_GPU=1` | the GPU belongs to someone else: every `device` row is `skipped` on both sides, and nothing touches the GPU (torch makes no `torch.cuda` call at all) |
| `CAJETA_BENCH_MACHINE`, `CAJETA_BENCH_RUN_ID` | the `machine` / `run_id` columns (default hostname, `<UTC>-<version>`) |
| `CAJETA_ML_BENCH_DATA` | the data directory (bench.sh sets it) |

## The rows

`area` is `ml` and `benchmark` the page slug, so every row id is `ml/<name>`.
The Cajeta rows have `platform` `dev.cajeta.ml 0.11.1`; the competitor rows
`scikit-learn 1.9.1` or `torch 2.14.1+cu130 cpu|cuda`, `language` `python`.

| id | variants | one iteration | check (against the competitor, same data) |
|---|---|---|---|
| `ml/linear-regression` | synthetic, housing | construct, fit on train, predict test | R² within 1e-9 of scikit-learn's |
| `ml/logistic-regression` | synthetic, digits | newton-cholesky, C = 1, tol 1e-4, 100 iterations; digits multinomial | accuracy within 0.005 |
| `ml/kmeans` | synthetic, digits | k = 10, k-means++, Lloyd to convergence | inertia within 1e-6 (synthetic) / 5% (digits), relative |
| `ml/knn` | synthetic | k = 5, brute force, fit + predict the queries | accuracy within 0.002 |
| `ml/pca` | synthetic | full-SVD fit, 20 components | explained-variance ratios within 1e-8, relative |
| `ml/standard-scaler` | synthetic | fit and transform | max \|z − z_ref\| ≤ 1e-10 (first 1000 rows) |
| `ml/mlp-train-step` | cpu, device | one Adam step (lr 1e-3), 784-512-10, batch 64, cross-entropy | loss at step 100 below loss at step 1 |
| `ml/mlp-infer` | cpu, device | eval forward, batch 64 | logits within 1e-4 of torch's |
| `ml/cnn-train-step` | cpu, device | one Adam step, SmallCnn on 1×28×28, batch 64 | loss decreases over 100 steps |
| `ml/cnn-infer` | cpu, device | eval forward | logits within 1e-4 |
| `ml/transformer-block-train` | cpu, device | one Adam step, pre-norm encoder block d 512, 8 heads, ff 2048, seq 128, batch 64, MSE | loss decreases over 100 steps |
| `ml/transformer-block-infer` | cpu, device | eval forward | outputs within 1e-3 |
| `ml/autodiff-overhead` | cpu | forward + backward of the MLP; notes carry backward/forward | every gradient within 1e-4 of torch autograd's |
| `ml/checkpoint-load` | cpu | read the MLP's torch `.pt` and import it, 100 times | weights equal to torch's, bit for bit |

Smoke sizes (`CAJETA_BENCH_SMOKE=1`): synthetic sets 1e4 × 20 instead of
1e5 × 100; kNN 1e4 training rows and 1e3 queries instead of 1e5 and 1e4; PCA 10
components (the smoke data has 20 features); SmallCnn width 8 instead of 16;
the encoder block d 64, 4 heads, ff 256, seq 16, batch 8; 20 training steps for
the loss check instead of 100; 10 checkpoint loads per iteration instead of
100; logistic-regression/digits trains on 360 of its 1437 training rows (the
multinomial Newton fit measured 72 s per fit at full size, 6 minutes of smoke).
The MLP, housing and the other digits rows are full size either way. Each row's
`dataset_name` and `model` name the sizes it ran.

**Shared weights.** The neural rows start from one set of weights:
`gen_data.py` builds a torch twin of each `dev.cajeta.ml.zoo` model
(`competitors/python/twins.py`, state_dict keys equal to the cajeta parameter
names), `torch.save`s its parameters, and the Cajeta side imports that file
with `PtReader.read` → `Checkpoints.torchToCajeta` → `importTorch` (strict).
That is what lets an inference row compare logits and the autodiff row compare
gradients.

**Device variants.** A `device` row runs with the model, the batch and the
optimizer state on the active backend (`dev.cajeta.ml` 0.11.1's device column;
the row's `backend` is the runtime's: hip, vulkan or cuda). It is skipped with
its reason in `notes` when `CAJETA_BENCH_NO_GPU=1` (`GPU held by another
session`) or when the library has no device MATMUL on this backend. torch's
device rows run on CUDA, or on ROCm from a ROCm wheel (backend `hip`), when the
GPU is free.

## Tolerances, and why

| check | tolerance | why |
|---|---|---|
| linear-regression R² | 1e-9 absolute | both are QR least squares in float64; they differ only in summation order |
| logistic-regression accuracy | 0.005 absolute | the same newton-cholesky objective and stopping rule; a row or two on the decision boundary may flip between two converged float64 solutions (0.005 is 1 of 200 test rows) |
| kmeans inertia, synthetic | 1e-6 relative | ten blobs far apart: every k-means++ seeding converges to the same partition, whatever its RNG, so only rounding remains |
| kmeans inertia, digits | 5% relative | overlapping clusters: the two libraries seed k-means++ from different RNG streams and land in different local optima; scikit-learn's own inertia spans 4.6% over `random_state` 0–19 (measured: 1.16519e6 to 1.21863e6) |
| knn accuracy | 0.002 absolute | brute force both sides; only equal-distance ties may break differently (sklearn's tie order is an implementation detail) |
| pca explained-variance ratio | 1e-8 relative | full SVD both sides on a spectrum with distinct eigenvalues |
| standard-scaler | 1e-10 absolute | ddof-0 mean and scale in float64; summation order only |
| mlp / cnn logits | 1e-4 absolute | float32; the 784- and 128-wide dot products accumulate in a different order |
| transformer output | 1e-3 absolute | float32 through two LayerNorms, softmax attention and GELU, with outputs of order 1–10 |
| autodiff gradients | 1e-4 absolute | float32 cross-entropy backward through the same two layers |
| training rows | loss decreases | the page's check: Adam on a fixed batch must lower the loss over 100 steps |
| checkpoint-load | bit-equal | a load is a copy; any difference is a bug |

The competitor rows make the same check against the same reference with the
same tolerance (their reference was computed by the same library, so theirs is
a sanity check of the pipeline; the Cajeta row's is the real one).

## Measured (Phoenix, 2026-10-25, smoke)

The smoke run (run id `20261025T1915Z-0.46.3`; `results/` is gitignored here,
because a smoke run is never committed, spec §1.5.1 and §9.3): `test/bench_smoke.sh` PASS in 206 s — data 5 s,
build a cache hit, Cajeta rows 191 s, competitor rows 7 s — at load average
15–21 from other sessions' compiles, so its timings are not numbers to quote.
Its slowest Cajeta rows: logistic-regression/digits 20.6 s per fit on 360
training rows (72 s on the full 1437; scikit-learn 0.17 s), knn 5.0 s per
1000 queries over 1e4 rows (scikit-learn 11 ms). Every check passed with room:
k-means/digits landed 1.06% from scikit-learn's inertia, every other check at
least four orders of magnitude inside its tolerance.

No full-size run yet. Scaling the smoke timings, kNN at 1e5 × 100 with 1e4
queries is about 40 minutes per iteration (500 times the smoke work; seven iterations) and the
transformer block at d 512, seq 128, batch 64 several minutes per step, so a
full run is hours; announce it with that estimate.

## Notes from building this

- `dev.cajeta.ml` is a version dependency (`0.11.1`, this checkout's
  `details.version`), resolved from `~/.olla` or Olla. A `{ "path": ".." }`
  dependency parses but resolves nothing on cajeta 0.46.3, so a working tree
  ahead of its release has to be `cajeta install`ed in `..` first.
- A build after any source change recompiles `dev.cajeta.ml` from its
  sources (3:01 to 3:31 on Phoenix, measured five times); an unchanged tree is
  a whole-artifact cache hit and takes seconds.
- Every benchmark class declares a constructor. On cajeta 0.46.3 a subclass
  with no constructor of its own, whose base declares one, comes out of `heap`
  unusable: the first field access faults (measured with a two-class probe;
  the same class with an empty explicit constructor works).
