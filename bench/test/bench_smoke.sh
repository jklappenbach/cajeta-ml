#!/usr/bin/env bash
# bench/test/bench_smoke.sh — the dev.cajeta.ml benchmark smoke
# (cajeta agents/benchmark-reference-plan.md 17.1.1 and 17.1.3).
#
# Runs ./bench.sh --competitors in smoke mode (CAJETA_BENCH_SMOKE=1: warmup 1,
# trials 3, reduced sizes) and then asserts, against the pages in the compiler
# repo's bench/ml/ (Classical.md, Neural.md):
#   * `--list` names exactly the page ids, ml/<name>;
#   * every id has a Cajeta row AND a competitor row (scikit-learn for the
#     classical page, PyTorch for the neural one) for every variant;
#   * every cpu / dataset variant row is `ok` with check_ok true on both sides:
#     classical metrics within the recorded tolerance of scikit-learn's on the
#     same data, training rows with a decreasing loss, inference rows with
#     logits within epsilon of torch's on the same weights;
#   * every `device` row is `skipped` while the GPU is held
#     (CAJETA_BENCH_NO_GPU=1), on both sides; with the GPU on, a torch device row
#     skipped because torch cannot run on this box's device is reported as a gap;
#   * both CSVs carry the harness's exact schema-v2 header, every row is
#     tagged smoke, and cajeta-bench's tools/verify-fidelity.py passes.
#
# The GPU on this box may belong to another session, so the smoke defaults to
# CAJETA_BENCH_NO_GPU=1; export CAJETA_BENCH_NO_GPU=0 to measure the device
# variants (they must then be ok, not skipped).
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bench="$(cd "$here/.." && pwd)"

export CAJETA_BENCH_SMOKE=1
export CAJETA_BENCH_NO_GPU="${CAJETA_BENCH_NO_GPU:-1}"

t0=$(date +%s)
echo ">> bench_smoke (dev.cajeta.ml): start $(date +%H:%M:%S), NO_GPU=$CAJETA_BENCH_NO_GPU"
log="$(mktemp)"
trap 'rm -f "$log"' EXIT

if [[ ! -x "$bench/bench.sh" ]]; then
    echo "FAIL: $bench/bench.sh is missing"
    exit 1
fi
"$bench/bench.sh" --competitors 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
run_dir="$(sed -n 's/^bench: results in //p' "$log" | tail -1)"
if [[ -z "$run_dir" || ! -f "$run_dir/results.csv" ]]; then
    echo "FAIL: bench.sh (exit $rc) wrote no results.csv"
    exit 1
fi

bin="$(sed -n 's/^bench: binary //p' "$log" | tail -1)"
list_out="$("$bin" --list 2>&1)" || { echo "FAIL: $bin --list exited non-zero"; exit 1; }

PY="${BENCH_PYTHON:-$bench/../../cajeta/bench/competitors/python/.venv/bin/python}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
"$PY" "$here/check_results.py" "$run_dir" "$list_out"
crc=$?

VERIFY="${VERIFY_FIDELITY:-$bench/../../cajeta-bench/tools/verify-fidelity.py}"
if [[ ! -f "$VERIFY" ]]; then
    echo "FAIL: verify-fidelity.py not found at $VERIFY (set VERIFY_FIDELITY)"
    exit 1
fi
"$PY" "$VERIFY" "$run_dir/results.csv"
vrc=$?

t1=$(date +%s)
echo ">> bench_smoke (dev.cajeta.ml): bench.sh exit $rc, checks exit $crc, verify-fidelity exit $vrc, $((t1 - t0)) s"
if [[ $rc -ne 0 || $crc -ne 0 || $vrc -ne 0 ]]; then
    echo "FAIL"
    exit 1
fi
echo "PASS"
