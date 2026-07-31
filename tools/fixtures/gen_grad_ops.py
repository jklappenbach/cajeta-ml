#!/usr/bin/env python3
# cajeta-ml v3 U2 — per-op forward + gradient pins from PINNED torch.
# Run: /home/julian/code/ml/venv-torch-ref/bin/python gen_grad_ops.py
# Inputs are integer-formula reconstructible in cajeta (float32).
import torch

print("torch", torch.__version__)
torch.set_printoptions(precision=10)

def a23():   # (2,3): (i*3+j+1)*0.5 - 1.75  -> mixed signs
    return torch.tensor([[(i * 3 + j + 1) * 0.5 - 1.75 for j in range(3)]
                         for i in range(2)], dtype=torch.float32)

def b23():   # (2,3): ((i+2*j) % 4) * 0.25 + 0.25  -> positive (safe div/log)
    return torch.tensor([[((i + 2 * j) % 4) * 0.25 + 0.25 for j in range(3)]
                         for i in range(2)], dtype=torch.float32)

def w32():   # (3,2): (i - j) * 0.5 + 0.25
    return torch.tensor([[(i - j) * 0.5 + 0.25 for j in range(2)]
                         for i in range(3)], dtype=torch.float32)

def r13():   # (1,3) broadcast row: j*0.5 + 0.5
    return torch.tensor([[j * 0.5 + 0.5 for j in range(3)]], dtype=torch.float32)

def flat(t):
    return [round(v, 8) for v in t.detach().reshape(-1).tolist()]

def pin(name, out, grads):
    print(f"== {name}")
    print("  out:", flat(out))
    for gname, g in grads:
        print(f"  d{gname}:", flat(g))

# Weighted loss so cotangents are non-uniform: L = sum(out * K), K = pos-grid
def K_like(t):
    f = t.reshape(-1)
    k = torch.tensor([((i % 3) + 1) * 0.5 for i in range(f.numel())],
                     dtype=torch.float32).reshape(t.shape)
    return k

def run(name, mk_inputs, fwd):
    ins = [t.clone().requires_grad_(True) for t in mk_inputs]
    out = fwd(*ins)
    L = (out * K_like(out)).sum()
    L.backward()
    pin(name, out, [(chr(ord('a') + i), t.grad) for i, t in enumerate(ins)])

run("add_bcast", [a23(), r13()], lambda a, b: a + b)
run("sub_bcast", [a23(), r13()], lambda a, b: a - b)
run("mul_bcast", [a23(), r13()], lambda a, b: a * b)
run("div_bcast", [a23(), r13()], lambda a, b: a / b)
run("matmul", [a23(), w32()], lambda a, b: a @ b)
run("relu", [a23()], torch.relu)
run("gelu", [a23()], torch.nn.functional.gelu)
run("tanh", [a23()], torch.tanh)
run("sigmoid", [a23()], torch.sigmoid)
run("softmax", [a23()], lambda a: torch.softmax(a, dim=-1))
run("logsoftmax", [a23()], lambda a: torch.log_softmax(a, dim=-1))
run("exp", [a23()], torch.exp)
run("log", [b23()], torch.log)
run("sqrt", [b23()], torch.sqrt)
run("pow3", [b23()], lambda a: a ** 3.0)
run("sum_full", [a23()], lambda a: a.sum().reshape(1))
run("mean_full", [a23()], lambda a: a.mean().reshape(1))
run("sum_axis0", [a23()], lambda a: a.sum(dim=0, keepdim=True))
run("mean_axis1", [a23()], lambda a: a.mean(dim=1, keepdim=True))
run("max_axis1", [a23()], lambda a: a.max(dim=1, keepdim=True).values)
run("reshape", [a23()], lambda a: a.reshape(3, 2))
run("transpose", [a23()], lambda a: a.t())
run("slice_ax1", [a23()], lambda a: a[:, 1:3])
run("concat_ax1", [a23(), b23()], lambda a, b: torch.cat([a, b], dim=1))

# Composite: 2-layer MLP, mean-MSE loss (2.3.1)
x = a23().requires_grad_(False)
w1 = w32().requires_grad_(True)                       # (3,2)
b1 = torch.tensor([0.1, -0.2], dtype=torch.float32).requires_grad_(True)
w2 = torch.tensor([[0.5], [-0.25]], dtype=torch.float32).requires_grad_(True)  # (2,1)
b2 = torch.tensor([0.05], dtype=torch.float32).requires_grad_(True)
t = torch.tensor([[0.5], [-0.5]], dtype=torch.float32)
h = torch.relu(x @ w1 + b1)
y = h @ w2 + b2
loss = ((y - t) ** 2).mean()
loss.backward()
print("== mlp2")
print("  loss:", round(loss.item(), 8))
print("  dw1:", flat(w1.grad))
print("  db1:", flat(b1.grad))
print("  dw2:", flat(w2.grad))
print("  db2:", flat(b2.grad))

# ── float64 pins (the generic-E tape's second column) ──────────────────
def flat64(t):
    return [f"{v:.17g}" for v in t.detach().reshape(-1).tolist()]

a = a23().double().requires_grad_(True)
out = torch.nn.functional.gelu(a)
L = (out * K_like(out).double()).sum()
L.backward()
print("== gelu64")
print("  out:", flat64(out))
print("  da:", flat64(a.grad))

x64 = a23().double()
w1 = w32().double().requires_grad_(True)
b1 = torch.tensor([0.1, -0.2], dtype=torch.float64).requires_grad_(True)
w2 = torch.tensor([[0.5], [-0.25]], dtype=torch.float64).requires_grad_(True)
b2 = torch.tensor([0.05], dtype=torch.float64).requires_grad_(True)
t64 = torch.tensor([[0.5], [-0.5]], dtype=torch.float64)
h = torch.relu(x64 @ w1 + b1)
y = h @ w2 + b2
loss = ((y - t64) ** 2).mean()
loss.backward()
print("== mlp2_64")
print("  loss:", f"{loss.item():.17g}")
print("  dw1:", flat64(w1.grad))
print("  db1:", flat64(b1.grad))
print("  dw2:", flat64(w2.grad))
print("  db2:", flat64(b2.grad))
