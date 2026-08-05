#!/usr/bin/env python3
# cajeta-ml v3 §13 U6 — end-to-end trained transformer pins from torch.
# Run: /home/julian/code/ml/venv-torch-ref/bin/python gen_transformer_train.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch", torch.__version__)
B, TQ, TK, E, H, FF = 2, 3, 4, 4, 2, 8


def grid(shape, off, scale=0.2):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor([(((f + off) % 5) * 0.5 - 1.0) * scale
                         for f in range(n)],
                        dtype=torch.float32).reshape(shape)


def sinusoidal(max_len, d):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    i2 = torch.arange(0, d, 2, dtype=torch.float32)
    div = torch.exp(i2 * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: d // 2])
    return pe


layer = nn.TransformerDecoderLayer(d_model=E, nhead=H, dim_feedforward=FF,
                                   dropout=0.0, batch_first=True)
with torch.no_grad():
    off = 1
    for _, p in layer.named_parameters():
        p.copy_(grid(p.shape, off))
        off += 3

tgt = grid((B, TQ, E), 0)
mem = grid((B, TK, E), 2)
target = grid((B, TQ, E), 5, 0.3)
causal = torch.triu(torch.full((TQ, TQ), float("-inf")), diagonal=1)
mem_pad = torch.tensor([[False] * 4, [False, False, False, True]])
pe = sinusoidal(TQ, E)

opt = torch.optim.Adam(layer.parameters(), lr=0.05)
for _ in range(10):
    opt.zero_grad()
    y = layer(tgt + pe, mem, tgt_mask=causal,
              memory_key_padding_mask=mem_pad)
    loss = F.mse_loss(y, target)
    loss.backward()
    opt.step()
final = F.mse_loss(layer(tgt + pe, mem, tgt_mask=causal,
                         memory_key_padding_mask=mem_pad), target).item()
print("transformer final loss", repr(final))
