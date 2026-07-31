#!/usr/bin/env python3
# cajeta-ml v3 U3 — structured-op forward + gradient pins from PINNED torch.
# Run: /home/julian/code/ml/venv-torch-ref/bin/python gen_grad_structured.py
#
# Every input is an integer formula over the FLAT index, so cajeta rebuilds
# the identical bits without shipping data files. Losses are the weighted
# scalar L = sum(out * K), K[f] = ((f % 3) + 1) * 0.5 — non-uniform
# cotangents, so a wrong VJP can't hide behind a symmetric seed.
import torch
import torch.nn.functional as F

print("torch", torch.__version__)


def grid(shape, fn):
    n = 1
    for d in shape:
        n *= d
    flat = [fn(f) for f in range(n)]
    return torch.tensor(flat, dtype=torch.float32).reshape(shape)


def x_(f):   return (f % 7) * 0.25 - 0.75
def w_(f):   return (f % 5) * 0.5 - 1.0
def b_(f):   return f * 0.25 - 0.25
def g_(f):   return (f % 4) * 0.5 + 0.5     # positive scale (norm weights)


def flat(t):
    return [round(v, 8) for v in t.detach().reshape(-1).tolist()]


def K_like(t):
    n = t.reshape(-1).numel()
    k = torch.tensor([((i % 3) + 1) * 0.5 for i in range(n)],
                     dtype=torch.float32).reshape(t.shape)
    return k


def pin(name, out, named_grads, extra=None):
    print(f"== {name}")
    print("  shape:", list(out.shape))
    print("  out:", flat(out))
    if extra:
        for k, v in extra:
            print(f"  {k}:", flat(v) if torch.is_tensor(v) else v)
    for gname, g in named_grads:
        print(f"  d{gname}:", flat(g))


def run(name, inputs, fwd, extra=None):
    """inputs: list of (name, tensor, requires_grad)."""
    ins = []
    for nm, t, rg in inputs:
        c = t.clone()
        if rg:
            c.requires_grad_(True)
        ins.append((nm, c, rg))
    out = fwd(*[c for _, c, _ in ins])
    L = (out * K_like(out)).sum()
    L.backward()
    pin(name, out,
        [(nm, c.grad) for nm, c, rg in ins if rg],
        extra=extra(*[c for _, c, _ in ins]) if extra else None)


# ── conv2d ────────────────────────────────────────────────────────────
# x (2,2,4,4), w (3,2,3,3), b (3,)
X = grid((2, 2, 4, 4), x_)
W = grid((3, 2, 3, 3), w_)
B = grid((3,), b_)

run("conv2d_s1p1", [("x", X, True), ("w", W, True), ("b", B, True)],
    lambda x, w, b: F.conv2d(x, w, b, stride=1, padding=1))
run("conv2d_s2p0", [("x", X, True), ("w", W, True), ("b", B, True)],
    lambda x, w, b: F.conv2d(x, w, b, stride=2, padding=0))

# ── pooling ───────────────────────────────────────────────────────────
run("maxpool2d_k2s2", [("x", X, True)],
    lambda x: F.max_pool2d(x, kernel_size=2, stride=2))
run("avgpool2d_k2s2", [("x", X, True)],
    lambda x: F.avg_pool2d(x, kernel_size=2, stride=2))
run("adaptiveavgpool2d_1", [("x", X, True)],
    lambda x: F.adaptive_avg_pool2d(x, (1, 1)))
run("adaptiveavgpool2d_2", [("x", X, True)],
    lambda x: F.adaptive_avg_pool2d(x, (2, 2)))

# ── batchNorm2d ───────────────────────────────────────────────────────
# Train mode: normalize with BATCH stats (biased var); running stats update
# with UNBIASED var and momentum. Eval mode: normalize with running stats.
GAM = grid((2,), g_)
BET = grid((2,), b_)
EPS = 1e-5

run("batchnorm2d_train", [("x", X, True), ("gamma", GAM, True), ("beta", BET, True)],
    lambda x, g, bb: F.batch_norm(x, None, None, g, bb, True, 0.1, EPS))

# running-stat update pin (momentum 0.1 from zeros/ones)
rm = torch.zeros(2)
rv = torch.ones(2)
_ = F.batch_norm(X.clone(), rm, rv, GAM.clone(), BET.clone(), True, 0.1, EPS)
print("== batchnorm2d_running")
print("  running_mean:", flat(rm))
print("  running_var:", flat(rv))

# eval mode against those updated running stats
RM = rm.clone()
RV = rv.clone()
run("batchnorm2d_eval", [("x", X, True), ("gamma", GAM, True), ("beta", BET, True)],
    lambda x, g, bb: F.batch_norm(x, RM, RV, g, bb, False, 0.1, EPS))

# ── layerNorm (last axis) ─────────────────────────────────────────────
LX = grid((2, 3, 4), x_)
LG = grid((4,), g_)
LB = grid((4,), b_)
run("layernorm_last", [("x", LX, True), ("gamma", LG, True), ("beta", LB, True)],
    lambda x, g, bb: F.layer_norm(x, (4,), g, bb, EPS))

# ── embedding gather ──────────────────────────────────────────────────
EW = grid((5, 3), x_)
IDX = torch.tensor([0, 2, 4, 1, 2], dtype=torch.long)   # index 2 repeats
run("embedding", [("w", EW, True)],
    lambda w: F.embedding(IDX, w))

# ── losses ────────────────────────────────────────────────────────────
# crossEntropy(logits (4,3), target class indices) — mean reduction.
LOGITS = grid((4, 3), x_)
TGT = torch.tensor([0, 2, 1, 2], dtype=torch.long)
lg = LOGITS.clone().requires_grad_(True)
ce = F.cross_entropy(lg, TGT, reduction="mean")
ce.backward()
print("== cross_entropy_mean")
print("  loss:", round(ce.item(), 8))
print("  dlogits:", flat(lg.grad))

# mse(pred, target) — mean reduction, gradient w.r.t. BOTH.
P = grid((2, 3), x_)
T = grid((2, 3), w_)
p = P.clone().requires_grad_(True)
t = T.clone().requires_grad_(True)
m = F.mse_loss(p, t, reduction="mean")
m.backward()
print("== mse_mean")
print("  loss:", round(m.item(), 8))
print("  dpred:", flat(p.grad))
print("  dtarget:", flat(t.grad))

# cosine similarity per row (SPELA's workhorse) + the 1-cos loss.
A = grid((3, 4), x_)
Bv = grid((3, 4), w_)
a = A.clone().requires_grad_(True)
bb = Bv.clone().requires_grad_(True)
cos = F.cosine_similarity(a, bb, dim=1, eps=1e-8)
print("== cosine_similarity")
print("  out:", flat(cos))
loss = (1.0 - cos).mean()
loss.backward()
print("== cosine_loss_mean")
print("  loss:", round(loss.item(), 8))
print("  da:", flat(a.grad))
print("  db:", flat(bb.grad))

# ── composite CNN block (3.3.1): conv → bn(train) → relu → maxpool ────
cx = X.clone().requires_grad_(True)
cw = W.clone().requires_grad_(True)
cb = B.clone().requires_grad_(True)
cg = grid((3,), g_).requires_grad_(True)
cbe = grid((3,), b_).requires_grad_(True)
h = F.conv2d(cx, cw, cb, stride=1, padding=1)
h = F.batch_norm(h, None, None, cg, cbe, True, 0.1, EPS)
h = F.relu(h)
h = F.max_pool2d(h, kernel_size=2, stride=2)
out = h
L = (out * K_like(out)).sum()
L.backward()
print("== cnn_block")
print("  shape:", list(out.shape))
print("  out:", flat(out))
print("  dx:", flat(cx.grad))
print("  dw:", flat(cw.grad))
print("  db:", flat(cb.grad))
print("  dgamma:", flat(cg.grad))
print("  dbeta:", flat(cbe.grad))
