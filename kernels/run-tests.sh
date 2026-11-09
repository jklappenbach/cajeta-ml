#!/usr/bin/env bash
# Build + run the dev.cajeta.ml.kernels suite.
#
# The suite lives under src/test/cajeta and is driven by cajeta-unit's reflective
# @Test discovery. The kernel library is built to a .cja and linked into the test
# binary with cajeta-unit; the backend the kernels are lowered for is XPU_BACKEND
# (cpu by default); the runtime picks the same one through CAJETA_XPU_BACKEND.
#
#   CAJETA     compiler binary (default: cajeta on PATH)
#   UNIT_REPO  cajeta-unit checkout (default: ../../cajeta-unit)
#   XPU_BACKEND  cpu | vulkan | amdgpu | nvptx
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
CAJETA="${CAJETA:-cajeta}"
XPU_BACKEND="${XPU_BACKEND:-${CAJETA_XPU_BACKEND:-cpu}}"
export CAJETA_XPU_BACKEND="${XPU_BACKEND}"
echo ">> backend: ${XPU_BACKEND}"
# A vulkan leg runs under the Khronos validation layer when it is installed: RADV
# segfaults on an invalid SPIR-V module; the layer names the instruction and the
# runtime refuses the module instead (cajeta benchmark-reference 14.3.7).
if [[ "${XPU_BACKEND}" == vulkan && -z "${VK_INSTANCE_LAYERS:-}" \
      && -f /usr/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json ]]; then
    export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
    echo ">> vulkan: VK_LAYER_KHRONOS_validation on"
fi

cajeta_artifact_path() {
    local dir="$1" name="$2"
    local p
    p=$( cd "$dir" 2>/dev/null && "$CAJETA" artifact-path 2>/dev/null ) || true
    if [[ -n "$p" && -f "$p" ]]; then printf '%s\n' "$p"; return 0; fi
    ls -t "$dir"/build/archive/"$name"-*.cja 2>/dev/null | head -1
}

UNIT_REPO="${UNIT_REPO:-$here/../../cajeta-unit}"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
OLLA_HOME="${OLLA_HOME:-$HOME/.olla}"
OLLA_URL="${OLLA_URL:-https://olla.cajeta.dev}"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1;
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

unit_cja="${UNIT_CJA:-}"
if [[ -z "$unit_cja" && -d "$UNIT_REPO" ]]; then
    echo ">> building cajeta-unit from checkout ($UNIT_REPO)"
    ( cd "$UNIT_REPO" && "$CAJETA" build >/dev/null )
    unit_cja="$(cajeta_artifact_path "$UNIT_REPO" dev.cajeta.unit 2>/dev/null)"
fi
if [[ -z "$unit_cja" ]]; then
    UNIT_VER="$(sed -n 's/.*"dev\.cajeta\.unit"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$here/cajeta.json" | head -1)"
    [[ -n "$UNIT_VER" ]] || { echo "no dev.cajeta.unit pin in cajeta.json" >&2; exit 1; }
    store_cja="$OLLA_HOME/dev.cajeta.unit/$UNIT_VER/dev.cajeta.unit-$UNIT_VER.cja"
    cache_cja="$here/build/.unit-cache/dev.cajeta.unit-$UNIT_VER.cja"
    if [[ -f "$store_cja" ]]; then unit_cja="$store_cja"
    elif [[ -f "$cache_cja" ]]; then unit_cja="$cache_cja"
    else
        echo ">> fetching dev.cajeta.unit $UNIT_VER from $OLLA_URL"
        meta="$(curl -fsS "$OLLA_URL/v2/resolve?name=dev.cajeta.unit&version=$UNIT_VER")"
        sha="$(printf '%s' "$meta" | sed -n 's/.*"sha256":"sha256:\([0-9a-f]*\)".*/\1/p')"
        [[ -n "$sha" ]] || { echo "/v2/resolve gave no sha256" >&2; exit 1; }
        mkdir -p "$(dirname "$cache_cja")"
        curl -fsS -o "$cache_cja" "$OLLA_URL/v2/blob/$sha"
        got="$(sha256_of "$cache_cja")"
        [[ "$got" == "$sha" ]] || { rm -f "$cache_cja"; echo "sha256 mismatch fetching unit" >&2; exit 1; }
        unit_cja="$cache_cja"
    fi
fi
[[ -f "$unit_cja" ]] || { echo "could not resolve a dev.cajeta.unit archive" >&2; exit 1; }
echo ">> cajeta-unit: $unit_cja"

echo ">> building the kernels library .cja"
"$CAJETA" --emit=cja -o "$out/kernels.cja" --classpath="$unit_cja" \
    dev.cajeta.ml.kernels.Kernels.version "$here/src/main/cajeta" "$out" >/dev/null

echo ">> building + running the test binary (${XPU_BACKEND})"
"$CAJETA" --emit=exe --profile=test --xpu-backend="${XPU_BACKEND}" \
    --classpath="$out/kernels.cja,$unit_cja" \
    -o "$out/kerneltests" \
    dev.cajeta.ml.kernels.selftest.TestMain.run "$here/src/test/cajeta" "$out" >/dev/null
mkdir -p "$here/build"
"$out/kerneltests" | tee "$here/build/test-${XPU_BACKEND}.log"
rc="${PIPESTATUS[0]}"
grep -P '^census-row\t' "$here/build/test-${XPU_BACKEND}.log" > "$here/build/census-${XPU_BACKEND}.tsv" || true
exit "$rc"
