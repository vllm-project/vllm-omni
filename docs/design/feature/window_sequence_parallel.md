# Window-Aligned Sequence Parallelism (SeedVR2)

## Motivation

SeedVR2's diffusion transformer (`NaDiT`) has no global attention: every layer
partitions the video tokens into 3D windows and attends only inside a window.
The window size is normalised against a fixed 720p reference area, so raising the
output resolution grows the *number* of windows, never their size.

Ulysses sequence parallelism exists to make global attention tractable by
all-to-all'ing the head dimension; with window-local attention there is nothing
to rescue and two activation-sized all-to-all exchanges per layer would be pure overhead.
Window-aligned SP instead assigns **whole windows** to ranks:

* every rank keeps **all** attention heads, so no head-count divisibility is
  required,
* attention inside a layer needs **zero** communication,
* the only communication is the re-shard between consecutive layers whose window
  layouts differ (regular <-> shifted), plus one small reduction per
  text-producing layer.

## Window layouts

`window_geometry.py` ports the reference windowing verbatim (Apache-2.0,
ByteDance SeedVR2). For a post-patch token grid `(T, H, W)`:

```text
scale   = sqrt(45 * 80 / (H * W))          # 720p reference area
wh, ww  = ceil(round(H * scale) / nh), ceil(round(W * scale) / nw)
wt      = ceil(min(T, 30) / nt)            # BYTEDANCE_MAX_TEMPORAL_WINDOW
```

The regular method tiles with `slice(it * wt, min((it + 1) * wt, T))`; the
shifted method moves every cut by half a window and **clips at the boundary**
(`max(..., 0)` / `min(..., size)`) -- it is not a cyclic `torch.roll`. Both
methods produce an exact partition of the grid; the shifted layout is genuinely
different, so an `A -> B -> A -> B` layer schedule needs both directions of
re-sharding.

Windows are enumerated `for iw ... for ih ... for it ...` (w outermost, t
innermost) and tokens inside a window are row-major in `(t, h, w)`. The
canonical token id is `(t * H + h) * W + w`, which is exactly what the reference
obtains by windowing `torch.arange`.

Consequences that matter (all verified against the reference):

* Window token counts are **ragged** (edge windows are clipped), so a shard must
  be balanced by token count, not by window count.
* The number of layout transitions follows the actual layer schedule. For the
  released 3B checkpoint (`num_layers=32`, alternating methods) there are 31
  inter-layer transitions, not 16; `ensure_layout(current, required)` derives
  them instead of assuming `layer % 2`.

## Planner

`window_sp.py` implements a deterministic LPT assignment:

```text
windows sorted by (-video_token_count, original_window_id)
each window is placed on the rank with the smallest (assigned_tokens, rank_id)
each rank then lists its windows in ascending original order
```

The cache key covers the post-patch token grid, the full window method and
parameters (via a SHA-1 geometry fingerprint), the geometry version, the **SP
world size** and the planner version, so a different degree can never reuse a
stale assignment. Caches are LRU-bounded.

## Plan A redistribution

At a layout boundary the activations must move from the source assignment to the
destination assignment. With `Z_src[r]` / `Z_dst[d]` the canonical token ids held
by each rank and `C[r, d]` the number of rows rank `d` needs from rank `r`:

* `input_split_sizes` of rank `r` is row `r` of `C`, `output_split_sizes` is
  column `r`,
* each peer segment is packed in **destination-local order**, so the receiver can
  reorder with a single `argsort` of the concatenated destination positions,
* `network_exchange_required` is computed from the global count matrix, so every
  rank takes the same branch; a rank that happens to exchange nothing never
  skips the collective alone.

The runtime is a variable-split `torch.distributed.all_to_all_single` over an
explicit process group (`window_parallel_size`), plus a local-permutation fast
path when the whole group stays in place (which is the SP=1 case). The first
version is deliberately synchronous: no extra streams, no per-layer
`cuda.synchronize()` or barrier "fixups", and no hidden all-gather.

## Text stream

Text is replicated on every rank and stays identical: each layer's attention
produces one text output per window, ranks sum their local window outputs,
one SUM all-reduce runs on the window-SP group, and every rank divides by the
**global** window count of the current layout.

```text
T_next = (sum_r sum_{w in Omega_r} T_w) / W_global
```

This is neither a token-weighted mean nor a mean of rank means; a rank with no
window contributes a zero tensor (with the *same dtype as its peers* -- a dtype
mismatch changes the collective and deadlocks NCCL, which is why the empty-rank
path is covered by a GPU test).

## Framework changes

Window-aligned SP is exposed as `window_parallel_size` in
`DiffusionParallelConfig`. It is mutually exclusive with
`ulysses_degree` / `ring_degree` / `allgather_degree`: the window group is the SP
group itself and the Ulysses / Ring / AllGather subgroups are degenerate
singletons, so no legacy accessor can mistake a window-SP run for a Ulysses run.
Everything else is SeedVR2-local (planner, routing, model integration), matching
the maintainer guidance on #7723 to keep the framework surface minimal.

## SeedVR2 integration

`SeedVR2WindowRuntime` owns, per request: the layer schedule, the entry
distribution, `ensure_layout` transitions, the text reduction and the final
reconstruction back to canonical token order. The DiT forward keeps only this
rank's video rows from the patch embedding onwards; the patch projection,
norms, ADA modulation, residual and MLP are all tokenwise, so they run locally.

The block never calls a parallel attention strategy: the shared `Attention`
layer is created with `skip_sequence_parallel=True`, and window attention is a
packed-varlen call over `[video window, replicated text]` segments (with a
grouped-SDPA fallback for backends without a packed-varlen entry point).

## Reference quirks reproduced on purpose

* `vid_out_ada` is declared with `layers=["out"]`, whose embedding regrouping is
  arithmetically incompatible with its 1-D parameters. The reference never uses
  that value: it asks the forward-wide cache for `emb_repeat_0_vid`, a key the
  first block already wrote, and therefore applies the *block* attention
  modulation together with the learned `out_scale` / `out_shift`. The port
  applies that effective modulation directly; changing it would break parity.
* `mm_layers=10` controls weight sharing only. Text is consumed and produced by
  all 32 blocks; only the last block freezes its text MLP path.
* RoPE positions are window-local: a video token at window-local `(dt, dh, dw)`
  uses axial indices `(text_len + dt, dh, dw)`, and every window restarts at 0.
* The rotary span is `3 * (rope_dim // 3)` (126 of 128 channels for the 3B), so
  the last two channels pass through unrotated.

## Checkpoint compatibility

The parameter layout of the port mirrors the reference, with one explicit
exception: the released checkpoint stores the per-block RoPE table as
`blocks.<i>.attn.rope.rope.freqs`, while this port registers it as
`blocks.<i>.attn.rope.freqs`. The validation loader normalizes exactly that
suffix (and only that suffix); every other key must already match.

> The validation loader explicitly normalizes the reference RoPE buffer suffix
> from `.rope.rope.freqs` to `.rope.freqs`. Checkpoint validation fails on any
> remaining missing or unexpected key after normalization.

Normalization is restricted and collision-checked: a key that is already
normalized is left untouched, and two source keys mapping onto the same target
key raise instead of silently overwriting a tensor. Loading then fails on any
remaining missing, unexpected or shape-mismatched key, so a checkpoint that does
not match the port cannot reach inference. Full 32-layer loading is the
acceptance path; truncating the block list is a development fixture and requires
an explicit flag.

This section describes the validation loader only. It is not a statement that
the serving loader (still to land with the SeedVR2 P0 integration) performs the
same conversion.

## Validation

| Layer | What it proves |
|---|---|
| CPU (`test_window_sp_plan.py`) | geometry parity against an independent windowing implementation, partition coverage, planner determinism/LPT oracle, A->B and B->A routing, round trips, empty ranks, metadata bounds, cache keys |
| GPU transport (`window_sp_worker.py --case transport`) | the real `all_to_all_single` moves rows bit-exactly through all 31 schedule transitions at SP=2/4, including ranks without windows |
| GPU toy block (`--case toy-block`) | four layers `A -> B -> A -> B` of joint video+text window attention plus the global text mean match a single-rank oracle to float64 round-off |
| Real model (`--case seedvr2`) | the released 3B checkpoint: SP=N output matches the SP=1 output of the same port within the frozen fixture tolerance, with per-rank memory and timing recorded |

`benchmarks/diffusion/benchmark_window_sp.py` reports the doc-level timing and
traffic breakdown (plan build, transitions, all-to-all, text reduction, peak
memory per rank).

## Integration limits

`window_parallel_size > 1` is reserved for SeedVR2's model-level window-SP path
in this change. It does not add SeedVR2 serving integration or a model-support
rejection gate for other diffusion pipelines; other pipelines must leave this
option at its default of 1. Fail-fast validation for unsupported pipelines is
deferred to the SeedVR2 serving integration, where the model identity is known
before expensive work starts.

The default (`window_parallel_size == 1`) and every existing SP configuration are
unchanged.

## Limitations

* Plan A only: every layout boundary re-shards the full activation. Plan B
  (halo exchange) is a follow-up and is expected to win at large sizes.
* The SeedVR2 VAE and the serving surface are P0 scope (#7723); this change
  carries the DiT and the SP path.
* The packed-varlen attention kernel needs a backend that accepts
  `cu_seqlens`; the grouped-SDPA path is the portable fallback and is what the
  pinned L20 environment exercises.
