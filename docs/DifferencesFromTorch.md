# Differences from PyTorch

Recognizable, not faithful. Where a torch convention encodes a genuine
mistake — or just Python — this library corrects it. The numerics themselves
are pinned against **torch 2.13.0+cpu** in the test suite: optimizer
trajectories, layer forwards and backwards, and whole training curves are
compared step by step, not just at convergence.

- **The tape is per-step and one-shot, so there is no `zero_grad`.** Every
  training step builds a fresh `GradTape`. This is the single largest
  behavioral difference and it is deliberate: forgetting `optimizer.zero_grad()`
  is the most common way a hand-written torch loop silently trains wrong —
  gradients accumulate, the model still "trains", and nothing errors. Here
  there is no state to forget to clear.

- **First-order only.** No double backward, no `create_graph`, no higher-order
  gradients. `backward` may be called once per tape. Torch supports building a
  graph of the backward pass; that machinery is not here.

- **`Linear.weight` is `(in, out)`**, not torch's `(out, in)`. Torch stores the
  transpose because `F.linear` computes `x @ Wᵀ`; storing what you multiply by
  is the less surprising choice. `Checkpoints.transposeLinearWeights(sd)`
  converts a torch-authored checkpoint. **This is the difference most likely to
  bite you**, because a mis-transposed weight is shape-compatible on any square
  layer and fails only elsewhere.

- **Parameter discovery walks OWN declared fields, not inherited ones.** Torch's
  `nn.Module.__setattr__` intercepts every assignment and registers parameters
  wherever they appear in the hierarchy. Here the walk is reflective over a
  class's own declared fields, so a subclass does not report a parameterized
  base's parameters. Compose rather than inherit when adding parameters —
  `LoraLinear` holds a `Linear` in a field for exactly this reason.

- **Every source of randomness takes an explicit seed.** No global RNG, no
  `torch.manual_seed`, no ambient state: layer constructors take `uint64 seed`,
  `Batches` takes a seed, `Dropout` takes a seed. Same seed ⇒ bit-identical
  weights and bit-identical shuffles, on any machine. This is what makes it
  possible to pin whole training trajectories rather than loose tolerances.

- **Freezing is explicit and honored in two places.** `setTrainable(false)` makes
  `Nets.gradsOf` yield zeros *and* makes optimizers skip the parameter. Both are
  required: with only the zero gradient, AdamW's decoupled weight decay would
  still move a weight you believe is frozen — a real torch footgun where
  `requires_grad=False` parameters left in an optimizer's param group continue
  to decay. Constructing a `LoraLinear` does **not** freeze its base; call
  `freezeBase()`.

- **Schedules are pure functions, not stateful objects.** `Schedules.cosineLr(base,
  tMax, step)` computes an LR from its arguments. Torch's `lr_scheduler` objects
  hold internal step counters, which is why `scheduler.step()` called in the
  wrong place (or the wrong number of times) silently shifts the whole curve.
  Here the LR is reproducible from `(base, step)` alone.

- **Unlabeled online learning is opt-in and counts what it skips.**
  `SpelaTrainer.observeUnlabeled` requires `cfg.selfDistill = true` and gates on
  `confidenceThreshold`; refused samples increment `skippedCount()`. Nothing
  pseudo-labels a sample the model is unsure about — the denominator stays
  honest. Torch has no equivalent because it has no equivalent trainer.

- **Mixed device residency is refused, not resolved.** `Ops` throws when one
  operand is device-resident and the other is not, naming the op. Torch inserts
  the copy, which is convenient right up until a per-op host↔device transfer in
  an inner loop becomes a performance mystery. (Today only the CPU column of the
  seam is implemented, so this is a contract being kept ahead of its backend.)

- **Errors throw typed exceptions** (`MlException`, `OptimizerException`), not
  warnings plus NaN propagation.

- **Two trainers, one shape.** `BackpropTrainer` is the ordinary global-backward
  loop. `SpelaTrainer` trains each layer against its own local objective, with
  no backward pass spanning the network — per-layer losses, per-layer freezing,
  truncatable inference. Torch has the first and nothing like the second.

## Not implemented

Absent, and honestly so — not "coming soon":

- **No GPU execution.** `ml.grad.Ops` is the device seam and its CPU column is
  complete; the GPU column is empty. No CUDA, no MPS, no distributed training.
- **No RNN / LSTM / GRU** layers, and no transformer *decoder* block (an
  encoder block and stack exist in `ml.zoo`).
- **No `torch.compile`, no TorchScript, no ONNX export.**
- **No autograd for higher-order derivatives, no functorch/vmap.**
- **No sparse tensors on the neural side** (`CsrMatrix` serves the classical
  linear models only).
- **No DataLoader ecosystem** — `Batches` covers seeded minibatching over
  in-memory tensors and nothing more: no workers, no samplers, no collate_fn.
- **Writing `.pt` is not supported** — read-only, by design. Save safetensors.
