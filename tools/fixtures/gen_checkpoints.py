#!/usr/bin/env python3
# cajeta-ml v3 U9 — checkpoint fixtures written by PINNED torch.
# Run: /home/julian/code/ml/venv-torch-ref/bin/python gen_checkpoints.py
#
# Writes into src/test/resources/checkpoints/. Values come from the same
# integer formulas the rest of the suite uses, so a reader can be checked
# against reconstructed expectations rather than a copied blob.
import json
import os
import struct

import torch
from safetensors.torch import save_file

OUT = os.path.join(os.path.dirname(__file__), "..", "..",
                   "src", "test", "resources", "checkpoints")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)
print("torch", torch.__version__, "->", OUT)


def grid(shape, fn):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor([fn(f) for f in range(n)],
                        dtype=torch.float32).reshape(shape)


def w_(f):  return (f % 5) * 0.5 - 1.0
def b_(f):  return f * 0.25 - 0.25


# ── a small MLP state_dict, cajeta's [in, out] weight orientation ─────
# Named to match what cajeta's Module walk produces for
# `Sequential(Linear(3,4), ReLU, Linear(4,2))` nested under `layers`.
state = {
    "layers.0.weight": grid((3, 4), w_),
    "layers.0.bias": grid((4,), b_),
    "layers.2.weight": grid((4, 2), w_),
    "layers.2.bias": grid((2,), b_),
}

save_file(state, os.path.join(OUT, "mlp_f32.safetensors"))
print("== mlp_f32.safetensors")
for k, v in state.items():
    print(f"  {k}: {list(v.shape)}")

# half / bfloat16 variants — both must widen to f32 on load
save_file({k: v.half() for k, v in state.items()},
          os.path.join(OUT, "mlp_f16.safetensors"))
save_file({k: v.bfloat16() for k, v in state.items()},
          os.path.join(OUT, "mlp_bf16.safetensors"))
print("== mlp_f16.safetensors / mlp_bf16.safetensors written")

# What bf16 rounding does to the values — the reader must reproduce
# exactly these, not the original f32.
bf = state["layers.0.weight"].bfloat16().float()
print("  bf16(layers.0.weight) ->",
      [round(v, 8) for v in bf.reshape(-1).tolist()])
hf = state["layers.0.bias"].half().float()
print("  f16(layers.0.bias) ->",
      [round(v, 8) for v in hf.reshape(-1).tolist()])

# ── the same state_dict as a torch .pt ────────────────────────────────
torch.save(state, os.path.join(OUT, "mlp_state.pt"))
print("== mlp_state.pt written")

# ── a hostile pickle: a .pt whose payload would EXECUTE on load ───────
# The constrained reader must refuse this, loudly. Built by hand so no
# dangerous object is ever constructed here either.
class _Evil:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


torch.save({"payload": _Evil()}, os.path.join(OUT, "evil.pt"))
print("== evil.pt written (must be REJECTED by the reader)")

# ── header shape of the f32 file, for the reader's own sanity ─────────
with open(os.path.join(OUT, "mlp_f32.safetensors"), "rb") as fh:
    raw = fh.read()
hlen = struct.unpack("<Q", raw[:8])[0]
hdr = json.loads(raw[8:8 + hlen])
print("== header")
print("  header_len:", hlen)
print("  keys:", sorted(k for k in hdr.keys() if k != "__metadata__"))
for k in sorted(hdr.keys()):
    if k != "__metadata__":
        print(f"  {k}: {hdr[k]}")
print("  total_bytes:", len(raw))
