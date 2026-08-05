#!/usr/bin/env python3
# cajeta-ml v3 §13 U6 — transfer-learning-with-preprocessing pins.
# Run: /home/julian/code/ml/venv-torch-ref/bin/python gen_transfer_aug.py
import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch", torch.__version__)


def grid(shape, off, scale=0.2):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor([(((f + off) % 5) * 0.5 - 1.0) * scale
                         for f in range(n)],
                        dtype=torch.float32).reshape(shape)


# (8, 1, 4, 4) images; normalization with EXPLICIT stats (the pipeline's
# deterministic member — random augments cannot pin across RNGs and are
# covered by their own unit).
imgs = grid((8, 1, 4, 4), 0, 0.25)
X = ((imgs - 0.1) / 0.8).reshape(8, 16)
Y = grid((8, 2), 3, 0.3)

l1 = nn.Linear(16, 4)
l2 = nn.Linear(4, 2)
with torch.no_grad():
    l1.weight.copy_(grid((4, 16), 1))
    l1.bias.copy_(grid((4,), 4))
    l2.weight.copy_(grid((2, 4), 7))
    l2.bias.copy_(grid((2,), 10))


def phase(params, steps):
    opt = torch.optim.Adam(params, lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(l2(F.relu(l1(X))), Y)
        loss.backward()
        opt.step()
    return F.mse_loss(l2(F.relu(l1(X))), Y).item()


la = phase(list(l1.parameters()) + list(l2.parameters()), 5)
print("phaseA", repr(la))
for p in l1.parameters():
    p.requires_grad = False
l2 = nn.Linear(4, 2)
with torch.no_grad():
    l2.weight.copy_(grid((2, 4), 50))
    l2.bias.copy_(grid((2,), 53))
lb = phase(list(l2.parameters()), 5)
print("phaseB", repr(lb))
for p in l1.parameters():
    p.requires_grad = True
lc = phase(list(l1.parameters()) + list(l2.parameters()), 5)
print("phaseC", repr(lc))
