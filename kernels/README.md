# dev.cajeta.ml.kernels

The GPU kernel package of cajeta-ml: compute primitives, dense linear algebra,
neural-network ops and the support to run them (tile families, tuning, test
tooling), on every `cajeta.xpu` backend. General purpose; cajeta-llm and
cajeta-xgboost are its first consumers.

- `dev.cajeta.ml.kernels` — the kernels and their host entry points: `Fill`,
  `Copy`, `Iota`, `Convert` and `Transpose` generic over dtype (spec §4.1,
  §4.10), `Reduce` (sum, min, max, sum of squares, whole or per segment, as a
  tree whose order never varies, and an exact sum in fixed point rounded once;
  spec §4.2), `Scan` (inclusive and exclusive prefix sums, plain or segmented,
  as reduce-then-scan; spec §4.3), `ArgMax` and `TopK` (per row, ties to the
  first index, k up to 64; spec §4.4), `Histogram` with `HistShared<TR>`
  (segmented integer histograms; the global-atomic variant and three shared-tile
  heights form a tile family timed at first use and remembered; spec §4.5),
  `Partition` (stable, segmented, by a predicate; spec §4.6), `Bucketize` (the
  upper-bound bucket against sorted edges; §4.7), `Gather` and `Scatter`
  (scatter-add included; §4.8), `RadixSort` (stable key and key-value sorts over
  int32 and float32 keys; §4.9), `Gemm` (f16, bf16 and f32 GEMM into float32 and int8
  into int32 through the matrix cores, any shape; §5.1), `MatVec` (decode
  mat-vec; §5.2), split-K with the split count chosen by measurement (§5.3), and a batched GEMM in one launch (§5.6), `Scale<TM>` (a tile family member), `gemm/DecodedGemm<D>` (a
  consumer-supplied decode step inside the tile skeleton).
- `dev.cajeta.ml.kernels.testing` — what a consumer's suite needs to account
  for every kernel it registers: `ValueCheck` (a launch is not a check),
  `KernelCensus` (run and value-checked, a probe, or a tracked skip, or the
  suite fails), `DeviceSkip` (a skip says why), `LaunchFailureBudget`.
- `dev.cajeta.ml.kernels.tune` — `TileShape`, `TileFamily<S>` (members pruned
  with a reason, the measured winner stored per problem, build and device) and
  `TuneMode`.
- `dev.cajeta.ml.kernels.mem` — `Arena<E>` (one block, views per step, reset)
  and `Scratch<E>` (grows only past its high-water mark): a step loop that
  allocates nothing after its first step.

```sh
./run-tests.sh                 # cpu backend
XPU_BACKEND=vulkan ./run-tests.sh
```
