# Noisy PP

> **Status:** experimental. This document describes the opt-in Noisy PP contract
> under `vllm_omni.experimental.ar_diffusion`. It is not a claim of general
> production support, and it does not define a public HTTP schema.
>
> **Tracking:** [JiusiServe/vllm-omni-project-manage#102](https://github.com/JiusiServe/vllm-omni-project-manage/issues/102)
> (feature, target v0.31). Model-quality follow-up:
> [#103](https://github.com/JiusiServe/vllm-omni-project-manage/issues/103)
> (target v0.32). Delivery PR:
> [vllm-project/vllm-omni#8282](https://github.com/vllm-project/vllm-omni/pull/8282).

## Table of Contents

- [Scope](#scope)
- [Overview](#overview)
- [Architecture](#architecture)
- [Scheduling](#scheduling)
- [NoisyKV](#noisykv)
- [Compatibility](#compatibility)
- [Configuration](#configuration)
- [Testing](#testing)
- [Limitations and open issues](#limitations-and-open-issues)
- [Related](#related)

## Scope

Noisy PP adds **same-request multi-chunk** scheduling on top of Omni's existing
layer pipeline parallel:

1. **Latest** (`Ordering.INTERLEAVED`): diagonal overlap across chunks; attention
   may read the newest finished noisy or clean KV version strictly earlier than
   the current slot.
2. **Serial** (`Ordering.SERIAL`): finish one chunk's stages before the next;
   history reads clean KV. Same topology as Latest; used as the Self-Forcing
   baseline.

Acceptance for #102 is **reproducible basic function** plus recorded run
config (deploy, `S·G`, `chunk_schedule`, history). This phase does **not**
promise FPS, speedup ratios, or realtime interaction SLAs. DreamZero / LingBot
default paged-session deploys must not regress.

## Overview

### What is Noisy PP?

Standard diffusion PP partitions one weight replica across `G` ranks and runs
every denoise step on that same set of ranks. Noisy PP places **`S` weight
replicas** on the same PP process group (`pipeline_parallel_size = S·G`):

- `S = 1`: native layer PP (no stage replicas).
- `S = T+1`: one stage per denoise step plus a final clean-KV stage.

Within a stage, ranks still own contiguous layer groups (`G = ⌈B/K⌉`). A chunk
walks the PP chain; activations move along `next_rank`, while versioned KV moves
**within a column** (same `g`, higher stage → lower stage) when Latest needs a
remote history page.

### Serial vs Latest

| Strategy | Chunk advance | History KV |
| --- | --- | --- |
| Serial | One chunk completes all stages before the next starts | Clean versions only |
| Latest | Chunks overlap on a diagonal schedule | Newest finished version with completion slot strictly earlier than the consumer |

Both strategies share one visibility rule: pick the highest step version whose
completion is strictly earlier than the current slot. Serial is the degenerate
schedule where “newest” is always clean.

`chunk_schedule` is an **in-generation** choice on one `ChunkPlan`. It is not a
session property. A session (optional) only decides whether clean KV survives
across generations; see [Realtime AR-Diffusion sessions](realtime_ar_diffusion.md).

### Reference path

WaveServe Wan 2.1 1.3B is the first opt-in integration:

- Pipeline: `WaveServeWanPipeline`
- Deploy: `vllm_omni/deploy/waveserve_wan.yaml`
- Recipe: `recipes/Physis-AI/WaveServe-Wan.md`
- Bench: `benchmarks/noisy_pp/waveserve_serial_vs_latest.py`

## Architecture

### Symbols

| Symbol | Meaning |
| --- | --- |
| `N`, `T` | Chunk count; denoise steps. Step `T` is the clean forward. |
| `S`, `G` | Stage count; layer groups per stage. Only `S ∈ {1, T+1}`. |
| `rank(s, g)` | `S=T+1` → `s·G + g`; `S=1` → `g`. |
| `(c, s)` | Task / KV version for chunk `c` at step `s` (plus `req` when batched). |
| Slot | Sync unit: one rank finishes its local `K` layers for one task. |
| `R` | Max micro-batch of requests on one rank in one slot. |

### Device layout

```text
s = pp_rank // G ,   g = pp_rank % G ,   rank(s, g) = s·G + g
```

```text
                 g = 0             g = 1                   g = G−1
stage 0     rank 0 ──hidden──▶ rank 1 ──▶ … ──▶ rank G−1 ──latent──▶
(step 0)       ▲ KV               ▲ KV                     ▲ KV
stage 1     rank G ──────────▶ … ─────────────────────▶ rank 2G−1
   ⋮
stage S−1   rank (S−1)G ─────▶ … ─────────────────────▶ rank SG−1 ──▶ gather
(clean)
```

Layer split for Noisy PP uses `(g, G)` (`get_pp_indices(B, pp_rank % G, G)`),
not a naive split over `world = S·G`. WaveServe builds Wan with explicit
`layer_pp_rank` / `layer_pp_world`; production Wan2.2 PP stays on its existing
`make_layers` path. Real Wan activations: non-last groups pack
`{latent, hidden_states}`; stage-last unpatches, advances with FlowEuler, and
packs `{latent}` to the next stage (or back to rank 0 when `S=1`).

### Components

```text
Optional Session (realtime tick / resident clean)
        │
        ▼
ARDiffusionModelRunner
  ├─ Legacy: SupportsARDiffusionPipeline → paged KV (DreamZero / LingBot)
  └─ Chunk:  SupportsARDiffusionChunkPipeline → NoisyKV + chunk plan
                    │
        ┌───────────┴────────────┐
        │ chunk_schedule         │ chunk_executor
        │  ChunkPlan / Ordering  │  slot loop, PP P2P, wait/evict
        └───────────┬────────────┘
                    ▼
              kv_cache/noisy.py  (VersionPool + transport)
```

| Piece | Owns | Does not own |
| --- | --- | --- |
| Runner | Pool bind, capacity, fail-closed cleanup | Denoise math, model conditioning |
| `chunk_schedule` | Serial/Latest plan, visibility, transfers, last-use | Tensor addresses, session identity |
| `chunk_executor` | Slot loop, activation P2P, KV exchange timing | Model-specific noise / CFG |
| NoisyKV | Version storage, peer transfer, eviction | Paged session sink/window policy |
| Pipeline | Capability declaration, adapter forward | Rank routing / LRU |

Pipelines may use the shared executor (WaveServe), keep a fully private loop
(DreamZero / LingBot default), or mix plan helpers with model hooks.

### Slot timeline (narrow wait)

For each slot on a rank:

1. Wait only for transfers that this slot's `sources` need.
2. Forward the local micro-batch; publish new versions.
3. Post KV `isend` for this slot's transfer plan (do not wait for remote recv).
4. Wait for this slot's posted handles, then `evict` versions whose last-use has
   passed and whose sends have completed.

Received versions become readable only on a later slot. Same-slot producer and
consumer of the same version are forbidden.

## Scheduling

Implemented in `experimental/ar_diffusion/chunk_schedule.py`.

### Inputs / outputs

- Input: `ChunkSchedule(chunks=N, num_denoise_steps=T, stages=S, layer_groups=G,
  ordering, kv_history_chunks=H)`.
- Output: `ChunkPlan` with per-slot `RankWork`, `sources`, `transfers`, and
  `last_use` for eviction.

### Visibility

For each history chunk and layer group, choose

```text
s* = max { s | P_g(c', s) < current_slot }
```

Clean version `(c', T)` is the highest step and stops upgrading once finished.
Serial schedules never expose unfinished noisy pages to later chunks.

### Invariants (selected)

- `S ∈ {1, T+1}`; intermediate stage counts are rejected.
- No Latest edge inside the same micro-batch / same slot on one rank.
- Cross-rank KV arrives before use without runtime negotiation.
- Finite `H` plus last-use eviction keeps per-rank residency bounded.

## NoisyKV

Implemented in `experimental/ar_diffusion/kv_cache/noisy.py`.

| Concern | Behavior |
| --- | --- |
| Unit | One version = local layer-group K/V for `(req, c, s)` |
| Storage | `VersionPool` sized at load from `ARDiffusionNoisyKVSpec` |
| Transport | Planned P2P on the existing PP group; column-local for Latest |
| Eviction | Scheduler `last_use`; no in-place overwrite of live send buffers |
| vs paged session KV | Separate pool today; runner still preallocates one or the other |

Resource sketch (per rank per slot, `S > 1`): KV traffic is on the order of
`⌊(H−1)/G⌋` versions × local layers × chunk tokens × heads × dim, often much
larger than activation traffic. Mitigations: post-on-produce, dedupe
`(version, dst)`, larger `G` (no cross-rank KV when `G ≥ H`), later quantization.

## Compatibility

Hard constraint: pipelines that only implement `SupportsARDiffusionPipeline`
keep today's bind / forward / stepwise behavior. Default DreamZero and LingBot
YAMLs stay on paged session paths and must not silently switch to
`run_chunk_pipeline`.

| Combination | Meaning | Status |
| --- | --- | --- |
| No session + Serial/Latest | WaveServe / Noisy PP offline | Supported (opt-in deploy) |
| Session + legacy loop | DreamZero / LingBot default | Supported (unchanged) |
| Session + Chunk Serial/Latest | Realtime + diagonal overlap | Designed; not required for #102 |

Capability surface: `SupportsARDiffusionChunkPipeline` with
`ar_diffusion_noisy_kv_spec()` and `bind_ar_diffusion_chunk_context()`.

## Configuration

| Layer | Key | Meaning |
| --- | --- | --- |
| Deploy `parallel_config` | `pipeline_parallel_size` | `S·G` |
| `ar_diffusion_stage_config` | `stage_parallel_size` | `S` |
| `ar_diffusion_stage_config` | `max_batch_size` | Micro-batch `R` |
| `ar_diffusion_stage_config` | `max_history_chunks` | Cap for `H` |
| Request `extra_args` | `chunk_schedule` | `serial` \| `latest` |
| Request `extra_args` | `num_chunks`, `num_denoise_steps`, `kv_history_chunks` | `N`, `T`, `H` |

Engine backend: `ARDiffusionEngine`. Use the mp executor from deploy YAML; do
not wrap Omni in `torchrun`. Keep `S = num_denoise_steps + 1` when overriding
PP size for `S > 1`.

## Testing

### CPU contracts

- `tests/diffusion/ar_diffusion/test_chunk_schedule.py`
- `tests/diffusion/ar_diffusion/test_chunk_executor.py`
- `tests/diffusion/ar_diffusion/test_noisy_kv.py`
- `tests/diffusion/ar_diffusion/test_noisy_kv_transport.py`
- `tests/diffusion/ar_diffusion/test_waveserve_chunk_layers.py`
- `tests/entrypoints/test_resolve_waveserve_wan_config.py`

### GPU smoke / E2E (record config)

- 1 GPU: `S=1`, real Wan weights, `schedule=serial`
- 2 GPU: `S=2` (`T=1`), real Wan weights, gather→VAE device pinning
- Multi-GPU: `S=T+1` Serial vs Latest via
  `benchmarks/noisy_pp/waveserve_serial_vs_latest.py`
- Recipe commands: `recipes/Physis-AI/WaveServe-Wan.md`

## Limitations and open issues

- Original paged-session path (DreamZero / LingBot) and Noisy PP (WaveServe /
  NoisyKV) stay separate: runner preallocates one or the other; no TP × Noisy PP
  composition yet (WaveServe DiT is PP-oriented).
- Noisy PP is not wired to session yet.
- Micro-batch packing in a slot (`R`): varlen vs pad; metadata envelope when a
  tick carries `N>1` chunks / multiple identities.

## Related

- [Realtime AR-Diffusion sessions](realtime_ar_diffusion.md)
- [AR-Diffusion pipeline capability](../ar_diffusion_pipeline_capability.md)
- [Pipeline Parallel](pipeline_parallel.md)
- [Diffusion Continuous Batching](diffusion_continuous_batching.md)
- Code: `vllm_omni/experimental/ar_diffusion/{chunk_schedule,chunk_executor}.py`,
  `kv_cache/noisy.py`, `diffusion/models/waveserve_wan/`
