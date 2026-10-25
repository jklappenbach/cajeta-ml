"""torch twins of the dev.cajeta.ml models the neural page measures.

Each twin's state_dict keys are the cajeta Module walk's parameter names, so a
checkpoint written here imports into the cajeta model through
Checkpoints.torchToCajeta (rank-2 `.weight` transposed: torch keeps Linear
weights [out, in], cajeta [in, out]) and Checkpoints.importTorch (strict).

  Mlp          dev.cajeta.ml.zoo.Mlp           layers.{0,2}.{weight,bias}
  SmallCnn     dev.cajeta.ml.zoo.SmallCnn      conv1 bn1 conv2 bn2 head
  EncoderBlock dev.cajeta.ml.zoo.EncoderBlock  norm1 attn.{wq,wk,wv,wo} norm2 ff1 ff2

BatchNorm running statistics are module state in cajeta, not parameters, so
the checkpoint the cajeta side imports carries parameters only
(`params_only`); a fresh model has running mean 0 and variance 1 on both
sides, which is what eval-mode inference reads.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, n_in, hidden, n_out):
        super().__init__()
        mods, prev = [], n_in
        for h in hidden:
            mods += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        mods.append(nn.Linear(prev, n_out))
        self.layers = nn.Sequential(*mods)

    def forward(self, x):
        return self.layers(x)


class SmallCnn(nn.Module):
    def __init__(self, channels, width, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, width, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(width, eps=1e-5, momentum=0.1)
        self.pool1 = nn.MaxPool2d(2, 2, 0)
        self.conv2 = nn.Conv2d(width, width * 2, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(width * 2, eps=1e-5, momentum=0.1)
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.head = nn.Linear(width * 2 * 4, classes)

    def forward(self, x):
        h = self.pool1(F.relu(self.bn1(self.conv1(x))))
        h = self.pool2(F.relu(self.bn2(self.conv2(h))))
        return self.head(torch.flatten(h, 1))


class Attn(nn.Module):
    """MultiheadAttention as cajeta writes it: four nn.Linear projections."""

    def __init__(self, embed, heads):
        super().__init__()
        self.wq = nn.Linear(embed, embed)
        self.wk = nn.Linear(embed, embed)
        self.wv = nn.Linear(embed, embed)
        self.wo = nn.Linear(embed, embed)
        self.heads = heads

    def forward(self, x):
        b, t, e = x.shape
        h, d = self.heads, e // self.heads

        def split(z):
            return z.reshape(b, t, h, d).permute(0, 2, 1, 3).reshape(b * h, t, d)

        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        att = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * (1.0 / math.sqrt(d)), dim=-1)
        ctx = torch.bmm(att, v).reshape(b, h, t, d).permute(0, 2, 1, 3).reshape(b, t, e)
        return self.wo(ctx)


class EncoderBlock(nn.Module):
    """Pre-norm: h = x + attn(norm1(x)); out = h + ff2(gelu(ff1(norm2(h))))."""

    def __init__(self, embed, heads, hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed, eps=1e-5)
        self.attn = Attn(embed, heads)
        self.norm2 = nn.LayerNorm(embed, eps=1e-5)
        self.ff1 = nn.Linear(embed, hidden)
        self.ff2 = nn.Linear(hidden, embed)

    def forward(self, x):
        h = x + self.attn(self.norm1(x))
        return h + self.ff2(F.gelu(self.ff1(self.norm2(h))))


def params_only(model):
    """The parameters as a plain dict (no OrderedDict metadata, no buffers),
    each its own contiguous storage — what PtReader's allowlist accepts."""
    return {k: v.detach().contiguous().clone() for k, v in model.named_parameters()}


def cajeta_layout(name, t):
    """A parameter in cajeta's orientation: rank-2 `.weight` transposed."""
    a = t.detach().cpu().numpy()
    if a.ndim == 2 and name.endswith(".weight"):
        a = a.T
    return a.copy()
