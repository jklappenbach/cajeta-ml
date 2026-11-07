# dev.cajeta.ml.kernels

The GPU kernel package of cajeta-ml: compute primitives, dense linear algebra,
neural-network ops and the support to run them (tile families, tuning, test
tooling), on every `cajeta.xpu` backend. General purpose; cajeta-llm and
cajeta-xgboost are its first consumers.

- `dev.cajeta.ml.kernels` — the kernels and their host entry points.
- `dev.cajeta.ml.kernels.testing` — what a consumer's suite needs to account
  for every kernel it registers: `ValueCheck` (a launch is not a check),
  `KernelCensus` (run and value-checked, a probe, or a tracked skip, or the
  suite fails), `DeviceSkip` (a skip says why), `LaunchFailureBudget`.

```sh
./run-tests.sh                 # cpu backend
XPU_BACKEND=vulkan ./run-tests.sh
```
