---
id: ml-checkpoints-lora
applies-to: [dev.cajeta.ml.io.Checkpoints, dev.cajeta.ml.io.StateDict, dev.cajeta.ml.io.Safetensors, dev.cajeta.ml.io.PtReader, dev.cajeta.ml.nn.LoraLinear]
title: ml.io + LoRA — loading torch checkpoints, saving safetensors, low-rank adapters
description: Interop with torch checkpoints — safetensors and .pt reading, the allowlisted unpickler, state-dict name matching and the Linear transpose, plus LoRA adapters on Linear and attention projections.
---

# ml.io + LoRA — checkpoints and adapters

## Formats

| Format | Read | Write | Notes |
|---|---|---|---|
| safetensors | `Safetensors.read(path)` | `Safetensors.write(path, sd)` | **prefer this** |
| torch `.pt` | `PtReader.read(path)` | — | ZIP + allowlisted pickle |

**Prefer safetensors, and prefer it for a security reason, not a taste one:**
reading it cannot execute anything. It is an 8-byte little-endian header
length, a JSON header of `name → {dtype, shape, data_offsets}`, then raw
little-endian tensor bytes. No code, no object graph, no pickle.

F32, F16 and BF16 all load and the narrow ones **widen to f32** (exact — every
f16/bf16 value is representable). Writing emits F32 only; the f32 round trip is
bit-stable.

### Reading `.pt` safely

A torch `.pt` is a ZIP holding a pickle. `Unpickler` runs an **allowlist**: only
the opcodes and globals needed to rebuild a tensor state dict are honored, and
anything else is **refused loudly rather than skipped**. `REDUCE` calls nothing.
A pickle that wants to import a module or construct an arbitrary class fails the
load; it does not silently produce a partial dict.

Entries are read through the ZIP **central directory**, not local headers —
torch writes local headers with the data-descriptor flag set and zero sizes, so
a local-header reader gets zero-length tensors. Compressed entries are
**refused**, not skipped, so a truncated model is an error rather than a quiet
accuracy loss.

## Loading into a module

```cajeta
ArrayList<String> missing = Checkpoints.load(model, "model.safetensors", strict);
StateDict sd = Checkpoints.saveStateDict(model);
Checkpoints.save(model, "out.safetensors");
```

`load` returns the names it could **not** match. With `strict=true` a mismatch
throws; with `strict=false` you get the list and decide — which is the mode you
want when loading a backbone into a model with a new head.

Matching is by `parameterNames()`, which are torch `state_dict` keys exactly
(`blocks.0.attn.wq.weight`). If names do not line up, the cause is almost always
a container held in an unnamed field — see `ml-nn-modules`.

**`Checkpoints.transposeLinearWeights(sd)`** exists because torch stores
`nn.Linear.weight` as `(out, in)` and this package stores `(in, out)`. Apply it
when adapting a torch-authored dense checkpoint. A silently un-transposed
weight is shape-compatible only when the layer is square — which is exactly how
this bites you late, on the one layer where it happens to fit.

## LoRA

Low-rank adapters: freeze a big pretrained weight, train a small `A·B` beside it.

```cajeta
LoraLinear ad = heap LoraLinear(in, out, /*rank=*/8, /*alpha=*/16.0f, seed);
ad.freezeBase();                       // ← REQUIRED; construction does not freeze

MultiheadAttention att = heap MultiheadAttention(embed, heads, /*rank=*/8, alpha, seed);
att.freezeBaseProjections();
```

- **Freezing is explicit.** A freshly constructed `LoraLinear` has a *trainable*
  base — `freezeBase()` is what makes it LoRA rather than an oddly parameterized
  full fine-tune. Skip it and you train the entire weight while believing you
  are training `rank` columns of it, at full optimizer-state cost, with nothing
  visibly wrong. Same for `MultiheadAttention.freezeBaseProjections()`.
- **Zero-init makes the adapter a no-op at step 0** — `B` starts at zero, so the
  adapted model is *exactly* the base model before training. A LoRA that changes
  outputs before it has learned anything is misconfigured.
- `merge()` folds `A·B` into the base weight, after which the adapter costs
  nothing at inference. **Double merge is refused**, loudly.
- `LoraLinear` COMPOSES a `Linear` (`baseLinear()`) rather than extending it,
  because the parameter walk enumerates own declared fields only.
- Once frozen, the base relies on freezing being honored in *both*
  `Nets.gradsOf` and the optimizer — see `ml-nn-modules`.

## Hazards

- **`.pt` loading is the risky path by construction.** Prefer safetensors for
  anything you did not produce yourself. The allowlist is a mitigation, not a
  license to load untrusted pickles.
- **`strict=false` returns names, and ignoring the return defeats it.** An
  unchecked non-strict load is indistinguishable from a successful one right up
  until the model predicts nonsense.
- **Buffers are not parameters.** BatchNorm running statistics travel in the
  state dict but are not trained; a load that matches parameters and misses
  buffers gives a model that is correct in `train()` and wrong in `eval()`.

## Foreign-weight import (0.7.0, spec §13.6)

```cajeta
StateDict raw = PtReader.read(path);              // .pt rides the CONSTRAINED unpickler
StateDict sd = Checkpoints.torchToCajeta(raw);    // fused in_proj -> wq/wk/wv, weights transposed
ImportReport rep = Checkpoints.importTorch(model, sd);       // STRICT default
ImportReport r2 = Checkpoints.importStateDict(model, sd, false);  // permissive
// r2.missing()    — model params the checkpoint lacked
// r2.unexpected() — checkpoint keys no param matched
```

- Reconciliation is BIDIRECTIONAL and nothing is silently dropped: a
  dropped key is a model that runs and is wrong. Strict throws naming
  the offending key; permissive reports both lists.
- Shape disagreements ALWAYS throw, naming both shapes — even
  permissive is permission to skip, not to reinterpret.
- `renamePrefix(sd, "encoder.", "backbone.")` handles key retargeting;
  `torchToCajeta` handles LAYOUT (torch `[out, in]` Linear weights,
  fused attention projections) in one pass — run it on RAW torch dicts
  only, never on an already-cajeta-layout dict.
- v3 imports `state_dict` content only (safetensors/.pt readers from
  §6); a HuggingFace-config reader is out of scope.
