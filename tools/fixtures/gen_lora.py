#!/usr/bin/env python3
# cajeta-ml v3 U10 — LoRA pins from PINNED torch.
# Run: /home/julian/code/ml/venv-torch-ref/bin/python gen_lora.py
#
# The standard adapter: y = xW + b + (xA)B * (alpha/rank), with B
# initialized to ZERO so the adapter starts as an exact no-op.
import torch

print("torch", torch.__version__)


def grid(shape, fn):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor([fn(f) for f in range(n)],
                        dtype=torch.float32).reshape(shape)


def x_(f):  return (f % 7) * 0.25 - 0.75
def w_(f):  return (f % 5) * 0.5 - 1.0
def a_(f):  return (f % 3) * 0.25 - 0.25
def b_(f):  return (f % 4) * 0.5 - 0.5


def flat(t):
    return [round(v, 8) for v in t.detach().reshape(-1).tolist()]


IN, OUT, RANK, BATCH = 4, 3, 2, 2
ALPHA = 4.0
SCALE = ALPHA / RANK          # 2.0

X = grid((BATCH, IN), x_)
W = grid((IN, OUT), w_)       # cajeta's [in, out] orientation
BIAS = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
A = grid((IN, RANK), a_)
B = grid((RANK, OUT), b_)

base = X @ W + BIAS
print("== base")
print("  out:", flat(base))

# adapter contribution
adapted = base + (X @ A @ B) * SCALE
print("== adapted")
print("  scale:", SCALE)
print("  out:", flat(adapted))

# merged: fold A·B·scale into W, then a plain linear must agree exactly
Wm = W + (A @ B) * SCALE
merged = X @ Wm + BIAS
print("== merged")
print("  weight:", flat(Wm))
print("  out:", flat(merged))
print("  max_abs_diff_vs_adapted:",
      float((merged - adapted).abs().max()))

# gradients w.r.t. A and B only (the base is frozen)
a = A.clone().requires_grad_(True)
b = B.clone().requires_grad_(True)
y = X @ W + BIAS + (X @ a @ b) * SCALE
K = torch.tensor([((i % 3) + 1) * 0.5 for i in range(y.numel())],
                 dtype=torch.float32).reshape(y.shape)
(y * K).sum().backward()
print("== grads")
print("  dA:", flat(a.grad))
print("  dB:", flat(b.grad))

# with B = 0 the adapter is a no-op — the property that makes a wrapped
# model identical to the original before any training
b0 = torch.zeros_like(B)
noop = X @ W + BIAS + (X @ A @ b0) * SCALE
print("== zero_init_is_noop")
print("  max_abs_diff:", float((noop - base).abs().max()))
