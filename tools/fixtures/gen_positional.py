#!/usr/bin/env python3
# cajeta-ml v3 §13 U1 — positional-encoding pins from PINNED torch.
# Run: /home/julian/code/ml/venv-torch-ref/bin/python gen_positional.py
import math

import torch

print("torch", torch.__version__)
torch.set_printoptions(precision=10)


def sinusoidal(max_len, d):
    # The canonical torch-tutorial formulation, generalized to odd d:
    # pe[pos, 2i]   = sin(pos * exp(-2i * ln(10000) / d))
    # pe[pos, 2i+1] = cos(pos * exp(-2i * ln(10000) / d))
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    i2 = torch.arange(0, d, 2, dtype=torch.float32)
    div = torch.exp(i2 * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    ncos = d // 2
    pe[:, 1::2] = torch.cos(pos * div[:ncos])
    return pe


print("== sinusoidal d=6 maxLen=4 (row-major)")
for v in sinusoidal(4, 6).flatten().tolist():
    print(f"  {v!r}")

print("== sinusoidal d=7 maxLen=4 (ODD dim, row-major)")
for v in sinusoidal(4, 7).flatten().tolist():
    print(f"  {v!r}")


# ── 1.1.3 compose with Embedding: emb[ids] + pe[0:T], batch=1 ────────
def grid(shape, fn):
    n = 1
    for d_ in shape:
        n *= d_
    return torch.tensor([fn(f) for f in range(n)],
                        dtype=torch.float32).reshape(shape)


table = grid((5, 6), lambda f: (f % 7) * 0.25 - 0.75)   # vocab=5, d=6
ids = torch.tensor([3, 1, 4])                            # T=3
composed = table[ids] + sinusoidal(4, 6)[:3]
print("== embedding+sinusoidal T=3 d=6 (row-major)")
for v in composed.flatten().tolist():
    print(f"  {v!r}")

# ── 1.1.2 learned PE: gradient reaches the table ─────────────────────
# loss = sum(x + pe[positions]) over batch B=2, T=3: d(loss)/d(pe[t]) = B
# for used rows, 0 for unused — analytic; torch confirms.
lp = torch.nn.Embedding(4, 6)
with torch.no_grad():
    lp.weight.copy_(grid((4, 6), lambda f: (f % 5) * 0.5 - 1.0))
x = grid((2, 3, 6), lambda f: (f % 3) * 0.1)
posids = torch.arange(3).unsqueeze(0).expand(2, 3)
y = x + lp(posids)
y.sum().backward()
print("== learned-pe grad rows (row-major, 4x6)")
for v in lp.weight.grad.flatten().tolist():
    print(f"  {v!r}")
