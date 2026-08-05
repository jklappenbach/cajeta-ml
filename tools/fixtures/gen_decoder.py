#!/usr/bin/env python3
# cajeta-ml v3 §13 U2 — decoder-layer pins from PINNED torch.
# Run: /home/julian/code/ml/venv-torch-ref/bin/python gen_decoder.py
import torch
import torch.nn as nn

print("torch", torch.__version__)
torch.set_printoptions(precision=10)

B, TQ, TK, E, H, FF = 2, 3, 4, 4, 2, 8


def grid(shape, off):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor([(((f + off) % 5) * 0.5 - 1.0) * 0.2
                         for f in range(n)],
                        dtype=torch.float32).reshape(shape)


layer = nn.TransformerDecoderLayer(d_model=E, nhead=H, dim_feedforward=FF,
                                   dropout=0.0, batch_first=True)
layer.eval()
with torch.no_grad():
    off = 1
    for name, p in layer.named_parameters():
        p.copy_(grid(p.shape, off))
        print(f"param {name} shape {list(p.shape)} off {off}")
        off += 3

tgt = grid((B, TQ, E), 0)          # decoder input
mem = grid((B, TK, E), 2)          # encoder output

# causal mask on tgt; memory key padding: batch 1's LAST key padded
causal = torch.triu(torch.full((TQ, TQ), float("-inf")), diagonal=1)
mem_pad = torch.tensor([[False, False, False, False],
                        [False, False, False, True]])

y = layer(tgt, mem, tgt_mask=causal, memory_key_padding_mask=mem_pad)
print("== decoder out (B,TQ,E row-major)")
for v in y.flatten().tolist():
    print(f"  {v!r}")

# tgt input values for the cajeta side
print("== tgt (row-major)")
for v in tgt.flatten().tolist():
    print(f"  {v!r}")
print("== mem (row-major)")
for v in mem.flatten().tolist():
    print(f"  {v!r}")
