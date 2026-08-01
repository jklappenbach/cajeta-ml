#!/usr/bin/env python3
# cajeta-ml v3 U5 — optimizer trajectories, LR schedules, and an
# end-to-end training run, from PINNED torch.
# Run: /home/julian/code/ml/venv-torch-ref/bin/python gen_optim_train.py
import math
import torch
import torch.nn.functional as F

print("torch", torch.__version__)


def flat(t):
    return [round(v, 8) for v in t.detach().reshape(-1).tolist()]


# ── optimizer trajectories ────────────────────────────────────────────
# One parameter of 4 elements, a DETERMINISTIC gradient sequence (no
# model in the loop, so the pin isolates the optimizer): at step s the
# gradient is g[i] = ((s + i) % 5) * 0.25 - 0.5.
def grad_at(step, n=4):
    return torch.tensor([((step + i) % 5) * 0.25 - 0.5 for i in range(n)],
                        dtype=torch.float32)


P0 = torch.tensor([0.5, -0.25, 1.0, -0.75], dtype=torch.float32)
STEPS = 5


def trajectory(name, make_opt):
    p = P0.clone().requires_grad_(True)
    opt = make_opt([p])
    print(f"== {name}")
    for s in range(STEPS):
        opt.zero_grad()
        p.grad = grad_at(s).clone()
        opt.step()
        print(f"  step{s}:", flat(p))


trajectory("sgd_plain",
           lambda ps: torch.optim.SGD(ps, lr=0.1))
trajectory("sgd_momentum",
           lambda ps: torch.optim.SGD(ps, lr=0.1, momentum=0.9))
trajectory("adam",
           lambda ps: torch.optim.Adam(ps, lr=0.1, betas=(0.9, 0.999), eps=1e-8))
trajectory("adamw",
           lambda ps: torch.optim.AdamW(ps, lr=0.1, betas=(0.9, 0.999),
                                        eps=1e-8, weight_decay=0.01))
# SGD with torch's L2 weight_decay (COUPLED — folded into the gradient),
# the contrast against AdamW's decoupled form.
trajectory("sgd_weight_decay",
           lambda ps: torch.optim.SGD(ps, lr=0.1, weight_decay=0.01))


# ── LR schedules ──────────────────────────────────────────────────────
print("== schedule_cosine")
base, tmax = 0.1, 10
print("  lrs:", [round(base * 0.5 * (1 + math.cos(math.pi * s / tmax)), 8)
                 for s in range(tmax + 1)])

print("== schedule_warmup_cosine")
warm, total = 3, 10
lrs = []
for s in range(total + 1):
    if s < warm:
        lrs.append(base * (s + 1) / warm)
    else:
        prog = (s - warm) / (total - warm)
        lrs.append(base * 0.5 * (1 + math.cos(math.pi * prog)))
print("  lrs:", [round(v, 8) for v in lrs])


# ── end-to-end training trajectory (5.3.1) ────────────────────────────
# The reference MLP, explicit weights, FULL-BATCH SGD for 20 steps on a
# fixed input/target — same data order, so the weights must agree.
def grid(shape, fn):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor([fn(f) for f in range(n)],
                        dtype=torch.float32).reshape(shape)


def x_(f):  return (f % 7) * 0.25 - 0.75
def w_(f):  return (f % 5) * 0.5 - 1.0


X = grid((4, 3), x_)
TGT = grid((4, 2), lambda f: (f % 3) * 0.5 - 0.5)
w1 = grid((3, 4), w_).requires_grad_(True)
b1 = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32).requires_grad_(True)
w2 = grid((4, 2), w_).requires_grad_(True)
b2 = torch.tensor([0.05, -0.05], dtype=torch.float32).requires_grad_(True)

opt = torch.optim.SGD([w1, b1, w2, b2], lr=0.1)
losses = []
for s in range(20):
    opt.zero_grad()
    y = torch.relu(X @ w1 + b1) @ w2 + b2
    loss = F.mse_loss(y, TGT)
    loss.backward()
    opt.step()
    losses.append(round(loss.item(), 8))
print("== mlp_sgd_trajectory")
print("  losses:", losses)
print("  w1_final:", flat(w1))
print("  b1_final:", flat(b1))
print("  w2_final:", flat(w2))
print("  b2_final:", flat(b2))


# ── gradient clipping ─────────────────────────────────────────────────
# clip_grad_norm_ scales ALL grads by maxNorm/totalNorm when the global
# L2 norm exceeds the cap — one shared factor, not per-tensor clipping.
g1 = torch.tensor([3.0, 4.0], dtype=torch.float32)
g2 = torch.tensor([12.0], dtype=torch.float32)
a = torch.zeros(2, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
a.grad = g1.clone()
b.grad = g2.clone()
total = torch.nn.utils.clip_grad_norm_([a, b], max_norm=5.0)
print("== clip_grad_norm")
print("  total_norm:", round(float(total), 8))
print("  g1:", flat(a.grad))
print("  g2:", flat(b.grad))
