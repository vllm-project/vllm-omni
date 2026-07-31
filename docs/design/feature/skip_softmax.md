# Skip-Softmax

Skip-Softmax is the sparse-attention mode of the `TRTLLM_ATTN` backend. Usage — the config keys
and how to pick an operating point — is in
[Diffusion Attention Backends](../../user_guide/diffusion/attention_backends.md). This page explains
the algorithm.

## Motivation

In a long attention row (a video DiT can have tens of thousands of keys), the softmax weight
concentrates on a small fraction of the keys; the rest receive near-zero weight and barely move the
output. Computing softmax and the value-weighted sum over those keys is wasted work. Skip-Softmax
detects, per block of keys, when a block cannot matter and skips its softmax and its value
multiply.

It is approximate: a skipped block still carries a small non-zero contribution, so the mode is
opt-in and off by default.

## The online-softmax pass

Attention is computed in a single streaming pass over the keys, in tiles of 128. Per query row the
kernel maintains three running values:

- `m` — the largest score seen so far,
- `l` — the running denominator `Σ exp(sⱼ − m)`,
- `O` — the running numerator `Σ exp(sⱼ − m)·vⱼ`,

and returns `O / l` at the end. For each key tile it computes the tile's scores `Q · K_jᵀ · scale`,
updates `m`, and accumulates that tile's contribution into `l` and `O`. Rescaling `l` and `O` when
`m` grows keeps the single pass numerically exact.

## The skip test

Once a tile's scores are known, its largest score `tile_max` is compared against the running
maximum:

```text
if exp(tile_max − running_max) < threshold:
    skip this tile          # do not compute its softmax weights or its P·V accumulation
```

`exp(tile_max − running_max)` is an upper bound on the softmax weight any key in the tile can
receive: if even the tile's best key is far below the current maximum, every key in the tile is
negligible, and both the softmax (the exponentials) and the `P·V` accumulation for that tile can be
dropped. The tile's contribution to `l` and `O` is simply skipped.

## What this bounds

Two properties of the test shape the achievable speedup:

- **`Q · K_jᵀ` always runs.** The test needs `tile_max`, which comes from the tile's scores, so the
  score matmul is never skipped — only the softmax and the `P·V` accumulation are. The score matmul
  and the value matmul are comparable in cost, so skipping every eligible tile removes at most
  roughly half the attention work; the kernel-level speedup is bounded well under 2×, not unbounded.

- **The decision is per tile, not per key.** A tile is skipped only when *all* of its keys are
  collectively negligible; a single important key keeps the whole tile. How many tiles actually
  qualify depends on the data and rounds down to tile granularity, so `target_sparsity` selects an
  operating point on a calibrated curve — it is not a promise that a fixed fraction of tiles is
  skipped.

## The threshold

The per-tile threshold is normalized by sequence length:

```text
threshold = factor / seqlen
```

`factor` comes from one of two sources:

- **Calibrated** — `factor = a · exp(b · target_sparsity)`, where `a` and `b` are fit per model (and
  per expert, for a multi-expert model) so that a given `target_sparsity` lands near that fraction of
  skipped tiles on the calibration data.
- **Direct** — `factor = skip_softmax.threshold · seqlen`, the calibration-free path; the user sets
  the threshold themselves.

## Timestep gating

`disabled_until_timestep = D` keeps the mode off during the early, high-noise denoise steps and
turns it on once the normalized timestep `t` drops to `t ≤ D` (`t` runs `1.0` → `0.0` over the
schedule). The early steps set the global structure of the output and their errors propagate through
every later step, so keeping them dense costs a few skipped-tile opportunities but protects fidelity.

`t` is the scheduler's own timestep divided by `num_train_timesteps`, published by the pipeline via
`DenoiseProgressMixin.record_denoise_step`. It is deliberately not derived from the step index,
because schedulers space their steps non-uniformly. A pipeline that does not publish a timestep
stays dense when `disabled_until_timestep` is set.
