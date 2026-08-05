#!/usr/bin/env python3
# cajeta-ml v3 §13 U4 — fine-tune-loop metric pins from PINNED torch.
# Run: /home/julian/code/ml/venv-torch-ref/bin/python gen_finetune.py
import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch", torch.__version__)
torch.set_printoptions(precision=10)


def grid(shape, off, scale=0.2):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor([(((f + off) % 5) * 0.5 - 1.0) * scale
                         for f in range(n)],
                        dtype=torch.float32).reshape(shape)


X = grid((8, 3), 0, 0.25)
Y = grid((8, 2), 3, 0.3)

l1 = nn.Linear(3, 4)
l2 = nn.Linear(4, 2)
with torch.no_grad():
    l1.weight.copy_(grid((4, 3), 1))
    l1.bias.copy_(grid((4,), 4))
    l2.weight.copy_(grid((2, 4), 7))
    l2.bias.copy_(grid((2,), 10))


def forward(x):
    return l2(F.relu(l1(x)))


def phase(params, steps):
    opt = torch.optim.Adam(params, lr=0.05)
    last = None
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(forward(X), Y)
        loss.backward()
        opt.step()
        last = loss.item()
    return F.mse_loss(forward(X), Y).item()


# Phase A: pretrain everything.
la = phase(list(l1.parameters()) + list(l2.parameters()), 5)
print("phaseA loss", repr(la))

# Phase B: freeze the backbone, REPLACE the head, train the head only
# with a fresh optimizer.
for p in l1.parameters():
    p.requires_grad = False
l2 = nn.Linear(4, 2)
with torch.no_grad():
    l2.weight.copy_(grid((2, 4), 50))
    l2.bias.copy_(grid((2,), 53))
lb = phase(list(l2.parameters()), 5)
print("phaseB loss", repr(lb))

# Phase C: unfreeze, fresh optimizer over everything.
for p in l1.parameters():
    p.requires_grad = True
lc = phase(list(l1.parameters()) + list(l2.parameters()), 5)
print("phaseC loss", repr(lc))
