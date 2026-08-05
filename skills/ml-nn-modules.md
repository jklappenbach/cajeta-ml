---
id: ml-nn-modules
applies-to: [dev.cajeta.ml.nn.Module, dev.cajeta.ml.nn.Parameter, dev.cajeta.ml.nn.Sequential, dev.cajeta.ml.nn.Linear, dev.cajeta.ml.nn.Conv2d, dev.cajeta.ml.nn.MultiheadAttention, dev.cajeta.ml.nn.BatchNorm2d, dev.cajeta.ml.nn.LayerNorm, dev.cajeta.ml.nn.Dropout, dev.cajeta.ml.nn.Embedding, dev.cajeta.ml.nn.Init, dev.cajeta.ml.nn.Nets, dev.cajeta.ml.nn.NetRegressor, dev.cajeta.ml.nn.NetClassifier, dev.cajeta.ml.zoo.Mlp, dev.cajeta.ml.zoo.SmallCnn, dev.cajeta.ml.zoo.EncoderBlock, dev.cajeta.ml.zoo.EncoderStack]
title: ml.nn — modules, layers, parameters (the torch.nn role)
description: Building networks — the Module contract and its reflection-based parameter walk, the layer catalogue, Sequential and the zoo, train/eval modes, freezing, seeded init, and the estimator adapters.
---

# ml.nn — modules and layers

The `torch.nn` role. A `Module` owns `Parameter` fields, declares a forward
pass over a `GradTape<float32>`, and composes into trees.

## The Module contract

```cajeta
public class Mlp extends Module {
    Linear fc1;                       // declared FIELDS are what the walk finds
    ReLU act;
    Linear fc2;

    public Mlp(int64 in, int64 hidden, int64 out, uint64 seed) {
        this.fc1 #= heap Linear(in, hidden, seed);
        this.act #= heap ReLU();
        this.fc2 #= heap Linear(hidden, out, seed + 1);
        return;
    }

    public GradTensor forward(GradTape<float32> tape, GradTensor x) {
        return this.fc2.forward(tape, this.act.forward(tape, this.fc1.forward(tape, x)));
    }
}
```

`parameters()` and `parameterNames()` walk the module tree by **reflection over
declared fields**. Two consequences worth knowing before you design a class:

- **Only OWN declared fields are enumerated, not inherited ones.** A subclass
  that adds parameters to a parameterized base will not report the base's. This
  is why `LoraLinear` *composes* a `Linear` in a field rather than extending it.
- A container of submodules must be held in a **named field** for names to come
  out torch-shaped. Holding children in a named `Sequential` field yields
  `blocks.0.attn.wq.weight`; holding them loose yields `0.attn.wq.weight`, and
  no checkpoint will match. `children()` is the hook a container overrides so
  the walk can descend.

Names match torch's `state_dict` keys exactly, which is what makes
`Checkpoints.load` work against a file torch wrote.

## Layer catalogue

| Kind | Layers |
|---|---|
| Dense | `Linear(in, out, seed)` · `LoraLinear(in, out, rank, alpha, seed)` |
| Convolutional | `Conv2d(inCh, outCh, kernel, stride, pad, seed)` |
| Pooling | `MaxPool2d` · `AvgPool2d` · `AdaptiveAvgPool2d(outH, outW)` |
| Normalization | `BatchNorm2d(channels)` · `LayerNorm(shape)` |
| Attention | `MultiheadAttention(embed, heads, seed)` — LoRA overload takes `rank` |
| Activations | `ReLU` · `GELU` · `Tanh` · `Sigmoid` · `Softmax` |
| Regularization | `Dropout(p, seed)` |
| Shape | `Flatten` |
| Lookup | `Embedding(numEmbeddings, dim, seed)` |
| Container | `Sequential` — `.add(#m)`, `.at(i)`, `.count()` |

Prebuilt in `ml.zoo`: `Mlp`, `SmallCnn`, `EncoderBlock`, `EncoderStack`.

## Modes, freezing, and inference

```cajeta
net.train();                        // Dropout samples; BatchNorm updates running stats
net.eval();                         // Dropout is identity; BatchNorm uses running stats
Tensor<float32> p = net.predict(x); // inference wrapper: builds a tape, no-grads it, discards it
```

`train()` / `eval()` propagate through the whole tree. `predict` is the
convenience path when you want an output and no gradients.

Freezing is per-parameter and honored in two places at once:

```cajeta
base.setTrainable(false);
```

`Nets.gradsOf` yields **zeros** for a frozen parameter, and every optimizer
**skips** non-trainable parameters. Both are required: a zero gradient alone
still lets AdamW's decoupled weight decay move a weight you believe is frozen.

## Initialization

`Init` implements the usual schemes (Xavier/Glorot, Kaiming/He, uniform,
normal), and **every layer constructor takes an explicit `uint64` seed**. There
is no global RNG. Same seed ⇒ bit-identical weights, which is what lets the
test suite pin whole training trajectories against torch.

## Estimator adapters

`NetRegressor(#net, epochs, lr)` and `NetClassifier(#net, classes, epochs, lr)`
wrap a module as a `Predictor`, so a network drops into `Pipeline`,
`Split.crossValScore`, and everything else in the `ml-protocol` skill. They
convert at the boundary: the protocol is `float64`, the network is `float32`.

## Hazards

- **Take ownership deliberately.** `Sequential.add(#m)` and the adapters take
  `#` transfer. A plain-formal → plain-field store is a borrow the constructor's
  own drop chain frees on exit; store owned fields with `#=`.
- **`parameters()` order is the walk order**, and optimizers pair it positionally
  with the gradient array from `Nets.gradsOf`. Do not reorder one without the
  other — build both from the same `Parameter[]`.
- **`predict` is not free.** It builds and discards a tape per call. In a hot
  loop, drive one tape yourself.

## Transformer completion (0.7.0, spec §13)

```cajeta
SinusoidalPositionalEncoding pe = heap SinusoidalPositionalEncoding(maxLen, d);
LearnedPositionalEncoding lp = heap LearnedPositionalEncoding(maxLen, d, seed);
// Both ADD to (B, T, E) — never concatenate; T > maxLen FAILS LOUDLY.

Tensor<float32> causal = Masks.causal(b, t);                 // additive, NEG blocks
Tensor<float32> pad = Masks.keyPadding(lengths, tq, tk);     // per-batch lengths
Tensor<float32> both = Masks.combine(causal, pad);
GradTensor y = attn.forwardMasked(tape, x, both);            // self + mask
GradTensor c = attn.forwardCross(tape, x, memory, pad);      // q from x, k/v memory
Tensor<float32> w = tape.valueOf(attn.lastAttentionNode());  // (B·H, Tq, Tk) weights

TransformerDecoderLayer dl = heap TransformerDecoderLayer(e, h, ff, seed);
GradTensor out = dl.decode(tape, x, memory, true, selfMask, true, memMask);
TransformerDecoder dec = heap TransformerDecoder(nLayers, e, h, ff, seed);
```

- Masks are ADDITIVE `(B, Tq, Tk)` float32: 0 attends, `Masks.NEG` blocks
  — `exp(NEG)` underflows to EXACTLY 0, so blocked weights and their
  gradient paths are exactly zero (asserted in the suite, not assumed).
- The decoder layer is post-norm with ReLU FFN and NO dropout (torch
  parity fixtures run `dropout=0`; compose `Dropout` outside if wanted).
- `lastAttentionNode()` is the structural-verification hook — check your
  causal mask's triangle instead of trusting the loss curve.
- Field names mirror torch (`self_attn`, `multihead_attn`, `linear1..2`,
  `norm1..3`) so checkpoint keys line up; the wq/wk/wv split under each
  block maps from torch's fused `in_proj` via
  `Checkpoints.torchToCajeta` (see `ml-checkpoints-lora`).
- NOT in scope: GRAPH NEURAL NETWORKS (dev.cajeta.graph is classical
  graph analysis, not GNNs) and CONTRASTIVE LEARNING (deferred, no
  consumer — spec §13.7).
