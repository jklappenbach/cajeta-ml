---
id: ml-training
applies-to: [dev.cajeta.ml.train.BackpropTrainer, dev.cajeta.ml.train.SpelaTrainer, dev.cajeta.ml.train.SpelaConfig, dev.cajeta.ml.train.TrainHistory, dev.cajeta.ml.train.Loss, dev.cajeta.ml.optim.Optimizer, dev.cajeta.ml.optim.SGD, dev.cajeta.ml.optim.Adam, dev.cajeta.ml.optim.AdamW, dev.cajeta.ml.optim.Schedules, dev.cajeta.ml.data.Batches]
title: ml.train + ml.optim — the two trainers (backprop and SPELA), optimizers, schedules, batching
description: Training loops — BackpropTrainer's global backward vs SpelaTrainer's layerwise local objectives, when each is the right choice, optimizers and LR schedules, seeded batching, gradient clipping, and SPELA's online observe/flush surface.
---

# ml.train + ml.optim — training

Two trainers with the same shape from the outside. That symmetry is the point
of the seam: you can swap them without restructuring your code.

## BackpropTrainer — the ordinary one

Forward the whole network, one global backward, one optimizer step per batch.

```cajeta
Optimizer opt = heap Adam(net.parameters(), 0.01f);
BackpropTrainer tr = heap BackpropTrainer(net, #opt, Loss.CROSS_ENTROPY, 1.0f);
Batches data = heap Batches(x, y, /*batchSize=*/32, /*seed=*/23L);
TrainHistory hist = tr.fit(data, /*epochs=*/12);

hist.finalLoss(); hist.lossAt(0); hist.epochs();
```

`clipNorm <= 0` disables clipping. `stepOn(xb, yb, ps)` is public, so you can
drive your own loop (a custom schedule, early stopping) without reimplementing
the step.

## SpelaTrainer — layerwise local objectives

SPELA trains **each layer against its own local objective** rather than
propagating one global error end to end. Every layer keeps class vectors and a
local loss; there is no single backward pass spanning the network.

```cajeta
SpelaConfig cfg = heap SpelaConfig(/*numClasses=*/10);
cfg.optimizer = SpelaConfig.ADAMW;
cfg.lossType  = SpelaConfig.COSFACE;
SpelaTrainer tr = heap SpelaTrainer(#layers, #cfg);
TrainHistory hist = tr.fit(data, epochs);

tr.lastLossAt(layer);            // per-LAYER loss — there is no single loss
tr.classVectorsAt(layer);
tr.accuracyFromLayer(x, y, k);   // what the net would score if truncated at k
```

**Choose SPELA when** you want per-layer supervision, layerwise freezing, or
online adaptation on a stream. **Choose backprop when** you want the standard
thing and a single loss curve to reason about. `SpelaConfig(numClasses)` gives
sane defaults; `paperExact()` switches to the reference paper's settings.

### The online surface

This is what SPELA has and backprop does not — adaptation without labels:

```cajeta
cfg.selfDistill = true;              // ← REQUIRED; the online path is OPT-IN
cfg.confidenceThreshold = 0.6;
cfg.onlineBufferSize = 100;
// ... then, after construction and some supervised training:
tr.observeUnlabeled(x);   // buffer a sample if the model is confident enough
tr.flush();               // apply the buffered updates
tr.observedCount(); tr.skippedCount(); tr.bufferedCount();
tr.confidenceOf(x);       // the gate observeUnlabeled applies
```

**`selfDistill` must be on.** Without it there is no objective for an unlabeled
sample, so `observeUnlabeled` has nothing to do and returns `false` — which is
indistinguishable, at the call site, from a sample that failed the confidence
gate. If every observation is refused, check this flag before the threshold.

Samples below `cfg.confidenceThreshold` are **skipped, not guessed** —
`skippedCount()` is the honest denominator. Buffering is governed by
`onlineBufferSize` / `onlineFlushEvery`.

The config transfers into the trainer (`#cfg`), so **the gate is fixed at
construction**. To compare thresholds, build two trainers; you cannot mutate the
config afterward (the compiler rejects the read as use-after-move).

Layerwise control:

```cajeta
tr.freezeBackbone();                    // adapt only the head
tr.setLayerTrainable(layer, false);
tr.predictFromLayer(x, fromLayer);
```

## Optimizers

| Optimizer | Constructor | Notes |
|---|---|---|
| `SGD` | `SGD(#params, lr, momentum)` | `momentum=0` is plain SGD |
| `Adam` | `Adam(#params, lr)` | torch defaults for betas/eps |
| `AdamW` | `AdamW(#params, lr, weightDecay)` | **decoupled** decay — not L2-in-the-gradient |

All take `#` ownership of the `Parameter[]`, expose `getLr`/`setLr`, and **skip
non-trainable parameters** (see the freezing note in `ml-nn-modules`).

## Schedules

`Schedules` is a set of pure functions — no scheduler object holding hidden
state, so an LR is reproducible from `(base, step)` alone:

```cajeta
opt.setLr(Schedules.warmupCosineLr(base, warmupSteps, totalSteps, step));
```

`stepLr(base, stepSize, gamma, step)` · `exponentialLr(base, gamma, step)` ·
`cosineLr(base, tMax, step)` · `warmupCosineLr(base, warmupSteps, tMax, step)`.

SPELA drives its own schedule internally — call `planSteps(total)` first so it
knows the horizon, then read `currentLr()`.

## Batching

`Batches(x, y, batchSize, seed)` — `reorder(epoch)` reshuffles deterministically
per epoch, so epoch *n* is the same permutation on every run and every machine.
`batchX(b)` / `batchY(b)` yield fresh owned tensors, and the row shape of each
sample is preserved (a CNN batch stays 4-D; it is not flattened to rank 2).

## Gradient clipping

`BackpropTrainer.clipByGlobalNorm(grads, maxNorm)` follows torch's
`clip_grad_norm_`: the **global** L2 norm across every parameter is compared to
the cap and all gradients scale by one shared factor. Returns the norm *before*
clipping — the number worth logging when a run diverges. Clipping each tensor
separately would change the update's direction, not just its length.

## Hazards

- **Optimizer ownership.** `BackpropTrainer(net, #opt, ...)` takes the
  optimizer; the module stays a borrow (callers keep using it afterward). An
  interface value stored into a plain field from a plain formal is a borrow the
  constructor's drop chain frees on exit — a use-after-free that can survive
  local testing and surface later.
- **A fresh tape per step is required**, not an optimization. See `ml-grad`.
- **`fit` reshuffles per epoch**; if you want a fixed order, drive `stepOn`
  yourself.
- **SPELA has no single loss.** `TrainHistory` records the aggregate; per-layer
  truth is `lastLossAt(layer)`. Reading the aggregate as "the" loss will
  mislead you about which layer stopped learning.

## Augmentation & the fine-tune loop (0.7.0, spec §13.4–§13.5)

```cajeta
AugmentPipeline pl = heap AugmentPipeline(seed);   // ONE seeded Philox stream
pl.add(heap RandomHorizontalFlip(0.5));
pl.add(heap RandomCrop((int64) 2));
pl.add(heap RandomRotation(15.0));
pl.add(heap Normalize(#mean, #std));               // EXPLICIT stats, never batch-inferred
Tensor<float32> batch = pl.apply(imgs);            // NCHW in, NCHW out
pl.eval();    // stochastic transforms go INERT; Normalize still applies
```

Fine-tune loop (§13.5): `backbone.setTrainable(false)` freezes a subtree
(provably unchanged after steps — the suite compares values);
`Sequential.replace(i, #newHead)` swaps the head with the backbone
BIT-IDENTICAL; on unfreeze the optimizers (SGD/Adam/AdamW) give the
parameter FRESH state — stale momentum applied on unfreeze is the bug
this library refuses to have. Freeze flags survive checkpoint
round-trips (state_dict carries values, not configuration).
