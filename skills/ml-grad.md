---
id: ml-grad
applies-to: [dev.cajeta.ml.grad.GradTape, dev.cajeta.ml.grad.GradTensor, dev.cajeta.ml.grad.Ops, dev.cajeta.ml.grad.Op]
title: ml.grad — the runtime tensor tape (define-by-run autodiff over Tensor<E>)
description: Reverse-mode autodiff for the neural stack — GradTape<E> lifecycle, the 41-op vocabulary, backward/gradOf, no-grad and detach, the Ops device seam, and why the tape is one-shot and per-step.
---

# ml.grad — define-by-run reverse-mode autodiff

`GradTape<E extends Floating>` records tensor operations as they execute and
replays them in reverse to accumulate gradients. `E` is `float32` (the whole
`ml.nn` stack) or `float64` (when you want the extra precision for a gradient
check).

This is **not** the stdlib's `cajeta.nucleo.autograd`. That one is a scalar
tape plus a compile-time `Grad`, is compiler-integrated, and stays where it is.
This one handles dynamic module trees, checkpoint-loaded nets, and per-layer
tapes — the things a neural framework needs.

## The lifecycle

```cajeta
GradTape<float32> tape = heap GradTape<float32>();
GradTensor x = tape.leaf(xb, false);        // input: no gradient wanted
GradTensor w = tape.leaf(weights, true);    // parameter: gradient wanted
GradTensor y = tape.relu(tape.matmul(x, w));
GradTensor loss = tape.mseLoss(y, tape.leaf(target, false));

float64 v = (float64) tape.valueOf(loss).flatGet(0);
tape.backward(loss);                        // seeds ones, replays in reverse
Tensor<float32> gw = tape.gradOf(w);        // w's cotangent
```

- `leaf(value, requiresGrad)` puts a tensor on the tape. `requiresGrad=false`
  inputs still flow forward; they just accumulate no cotangent.
- `valueOf(node)` is the forward value, `gradOf(node)` the gradient — the
  latter only after `backward`.
- `backward(out)` seeds `out`'s cotangent with ones and walks nodes in reverse
  execution order.

## The tape is ONE-SHOT and PER-STEP — this is the design, not a limitation

Build a **fresh tape every training step**. That is what makes "zero the
gradients" unnecessary: there is no stale gradient state to forget to clear,
which is the single most common way a hand-written torch loop silently trains
wrong. `BackpropTrainer.stepOn` does exactly this, and so should your own loop.

Do not call `backward` twice on one tape, and do not keep a tape across steps
to save allocation. It is first-order by construction — no double backward, no
`create_graph`, no higher-order gradients.

## Op vocabulary (41 ops)

| Group | Ops |
|---|---|
| Elementwise binary | `add` `sub` `mul` `div` |
| Matmul | `matmul` (2-D, the tuned kernel) · `matmulBatched` · `transposeBatch` |
| Activations | `relu` `gelu` `tanh` `sigmoid` `softmax` `logSoftmax` |
| Unary math | `exp` `log` `sqrt` `neg` `powScalar` |
| Reductions | `sum` `mean` `sumAxis` `meanAxis` `maxAxis` `rowNorm` |
| Shape | `reshape` `transpose2d` `permute` `slice` `concat` |
| Structured | `conv2d` `maxPool2d` `avgPool2d` `adaptiveAvgPool2d` |
| Normalization | `batchNorm2dTrain` `batchNorm2dEval` `layerNorm` |
| Lookup / regularization | `embedding` `dropout` |
| Losses | `crossEntropy` `mseLoss` `cosineSim` |

Broadcast-widened elementwise ops restore input shapes on the backward pass via
`Tensor.sumTo` (PyTorch's `sum_to`), so `(n,k) + (k,)` differentiates correctly.

## No-grad and detach

```cajeta
tape.noGradBegin();
GradTensor scores = model.forward(tape, x);   // recorded, but no backward edges
tape.noGradEnd();

GradTensor stopped = tape.detached(node);     // value carries on, gradient stops
```

`noGradBegin/End` bracket a region for inference-inside-training (validation
scoring mid-epoch). `detached` is the per-node form — the stop-gradient used
by SPELA's layerwise targets and by self-distillation.

## Ops — the device seam

Every FORWARD kernel routes through `Ops`, which is the single place a device
backend plugs in. Today the CPU column is complete and the GPU column is empty;
`ml.grad` calls no tensor kernel directly, so adding a device does not touch
`GradTape`.

`Ops` refuses **mixed residency** loudly rather than silently copying: if one
operand is device-resident and the other is not, you get an exception naming
the op. A silent host↔device copy per op is how a framework becomes slow in a
way nobody can profile.

## Hazards

- **`gradOf` on a node whose `requiresGrad` was false** yields zeros, not an
  error — that is deliberate (a frozen LoRA base legitimately has no gradient),
  but it means a typo'd `false` shows up as a model that will not learn rather
  than as a crash. If training does nothing, check the leaf flags first.
- **`float64` tapes exist for gradient checking**, not for training. The whole
  `ml.nn` layer surface is `float32`; a `GradTape<float64>` cannot drive a
  `Module`.
- The tape holds its forward values and saved state for the backward pass, so
  peak memory scales with graph depth × activation size. A per-step tape is
  freed at the end of the step; a tape you hold onto is not.
