#!/usr/bin/env python3
# cajeta-ml v3 U4 — module-tree pins from PINNED torch.
# Run: /home/julian/code/ml/venv-torch-ref/bin/python gen_nn_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch", torch.__version__)


# ── 4.1.1 the checkpoint key contract ─────────────────────────────────
# A tree shaped like cajeta's: named sub-module fields + an indexed
# container. cajeta's Linear stores weight as [in, out] (torch is
# [out, in]) — the KEYS are what must agree, not the layouts.
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(4, 4)
        self.wo = nn.Linear(4, 4)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(3, 4)
        self.layers = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        self.block = Block()
        self.norm = nn.LayerNorm(2)


print("== state_dict_keys")
for k in Net().state_dict().keys():
    print("  " + k)

print("== cnn_state_dict_keys")


class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, 3, padding=1)
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Linear(12, 2)


for k in Cnn().state_dict().keys():
    print("  " + k)


# ── 4.3.1 reference MLP, explicit weights, forward + backward ─────────
def grid(shape, fn):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor([fn(f) for f in range(n)],
                        dtype=torch.float32).reshape(shape)


def x_(f):  return (f % 7) * 0.25 - 0.75
def w_(f):  return (f % 5) * 0.5 - 1.0


def flat(t):
    return [round(v, 8) for v in t.detach().reshape(-1).tolist()]


# cajeta: h = relu(x @ W1 + b1); y = h @ W2 + b2, W stored [in, out].
# torch's functional linear takes [out, in], so transpose on the way in.
X = grid((2, 3), x_)
W1 = grid((3, 4), w_)        # [in, out]
B1 = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32)
W2 = grid((4, 2), w_)        # [in, out]
B2 = torch.tensor([0.05, -0.05], dtype=torch.float32)
TGT = torch.tensor([[0.5, -0.5], [-0.25, 0.75]], dtype=torch.float32)

w1 = W1.clone().requires_grad_(True)
b1 = B1.clone().requires_grad_(True)
w2 = W2.clone().requires_grad_(True)
b2 = B2.clone().requires_grad_(True)
h = torch.relu(X @ w1 + b1)
y = h @ w2 + b2
loss = F.mse_loss(y, TGT, reduction="mean")
loss.backward()
print("== mlp_ref")
print("  y:", flat(y))
print("  loss:", round(loss.item(), 8))
print("  dw1:", flat(w1.grad))
print("  db1:", flat(b1.grad))
print("  dw2:", flat(w2.grad))
print("  db2:", flat(b2.grad))


# ── attention block, explicit weights ─────────────────────────────────
# B=2, T=3, E=4, H=2 -> D=2. Same math as cajeta's block:
# q,k,v = x@Wq..., heads split, softmax(qk^T/sqrt(D)) @ v, merge, @ Wo.
B, T, E, H = 2, 3, 4, 2
D = E // H
XA = grid((B, T, E), x_)
WQ = grid((E, E), w_)
WK = grid((E, E), lambda f: w_(f + 1))
WV = grid((E, E), lambda f: w_(f + 2))
WO = grid((E, E), lambda f: w_(f + 3))
BQ = torch.zeros(E)


def split(t):
    return t.reshape(B, T, H, D).permute(0, 2, 1, 3).reshape(B * H, T, D)


xa = XA.clone().requires_grad_(True)
wq = WQ.clone().requires_grad_(True)
wk = WK.clone().requires_grad_(True)
wv = WV.clone().requires_grad_(True)
wo = WO.clone().requires_grad_(True)

flat_x = xa.reshape(B * T, E)
q = split(flat_x @ wq)
k = split(flat_x @ wk)
v = split(flat_x @ wv)
scores = torch.bmm(q, k.transpose(1, 2)) * (1.0 / (D ** 0.5))
attn = torch.softmax(scores, dim=-1)
ctx = torch.bmm(attn, v)
merged = ctx.reshape(B, H, T, D).permute(0, 2, 1, 3).reshape(B * T, E)
out = (merged @ wo).reshape(B, T, E)
K = torch.tensor([((i % 3) + 1) * 0.5 for i in range(out.numel())],
                 dtype=torch.float32).reshape(out.shape)
L = (out * K).sum()
L.backward()
print("== attention")
print("  shape:", list(out.shape))
print("  out:", flat(out))
print("  dx:", flat(xa.grad))
print("  dwq:", flat(wq.grad))
print("  dwo:", flat(wo.grad))


# ── U6: pre-norm transformer encoder block ────────────────────────────
# h   = x + attn(norm1(x));  out = h + ff2(gelu(ff1(norm2(h))))
# Weights explicit so the cajeta block is directly comparable. cajeta
# stores Linear weight [in, out]; torch's functional form wants the same
# orientation here because we write `x @ W` by hand.
B2, T2, E2, H2, FF = 2, 3, 4, 2, 8
D2 = E2 // H2
XB = grid((B2, T2, E2), x_)
WQ2 = grid((E2, E2), w_)
WK2 = grid((E2, E2), lambda f: w_(f + 1))
WV2 = grid((E2, E2), lambda f: w_(f + 2))
WO2 = grid((E2, E2), lambda f: w_(f + 3))
F1 = grid((E2, FF), lambda f: w_(f + 1))
F2 = grid((FF, E2), lambda f: w_(f + 2))
G1 = torch.ones(E2)
B1_ = torch.zeros(E2)


def attn_fwd(xin):
    fl = xin.reshape(B2 * T2, E2)

    def sp(t):
        return t.reshape(B2, T2, H2, D2).permute(0, 2, 1, 3).reshape(B2 * H2, T2, D2)

    q, k, v = sp(fl @ WQ2), sp(fl @ WK2), sp(fl @ WV2)
    sc = torch.bmm(q, k.transpose(1, 2)) * (1.0 / (D2 ** 0.5))
    at = torch.softmax(sc, dim=-1)
    cx = torch.bmm(at, v)
    mg = cx.reshape(B2, H2, T2, D2).permute(0, 2, 1, 3).reshape(B2 * T2, E2)
    return (mg @ WO2).reshape(B2, T2, E2)


xb = XB.clone().requires_grad_(True)
n1 = F.layer_norm(xb, (E2,), G1, B1_, 1e-5)
h = xb + attn_fwd(n1)
n2 = F.layer_norm(h, (E2,), G1, B1_, 1e-5)
ffn = (F.gelu(n2.reshape(B2 * T2, E2) @ F1) @ F2).reshape(B2, T2, E2)
outb = h + ffn
Kb = torch.tensor([((i % 3) + 1) * 0.5 for i in range(outb.numel())],
                  dtype=torch.float32).reshape(outb.shape)
(outb * Kb).sum().backward()
print("== encoder_block")
print("  shape:", list(outb.shape))
print("  out:", flat(outb))
print("  dx:", flat(xb.grad))
