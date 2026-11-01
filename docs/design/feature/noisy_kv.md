# Noisy KV

> **Status:** experimental (`vllm_omni.experimental.ar_diffusion`). Not a public
> HTTP schema.
>
> **Tracking:** [#102](https://github.com/JiusiServe/vllm-omni-project-manage/issues/102)
> (v0.31), quality [#103](https://github.com/JiusiServe/vllm-omni-project-manage/issues/103)
> (v0.32). PR [#8282](https://github.com/vllm-project/vllm-omni/pull/8282).

## Table of Contents

- [Overview](#overview)
    - [Symbols](#symbols)
    - [What is a cell?](#what-is-a-cell)
- [Scope](#scope)
- [Architecture](#architecture)
- [Cell schedule](#cell-schedule)
- [Cell executor](#cell-executor)
- [NoisyKV](#noisykv)
    - [Lifecycle contract](#lifecycle-contract)
    - [Session integration (P1 target contract)](#session-integration-p1-target-contract)
- [Configuration](#configuration)
- [Compatibility](#compatibility)
- [Testing](#testing)
- [Open issues](#open-issues)
- [Acceptance cases (P1: CausVid, CPU-first)](#acceptance-cases-p1-causvid-cpu-first)
- [Related](#related)

## Overview

Noisy KV is the same-request multi-chunk path on Omni's existing **layer PP**.

### Symbols

| Symbol | Config | Meaning |
| --- | --- | --- |
| `N` | `num_chunks` | Chunks |
| `D` | `num_denoise_steps` | Noisy denoise steps only (not counting clean) |
| `S` | `step_replicas` | Step replicas on the denoise axis; `S ∈ {1, D, D+1}` |
| `G` | `layer_groups` | Layer groups per step replica; `world = S·G` |
| `B`, `K` | `blocks`, `blocks_per_rank` | DiT blocks; contiguous blocks per rank in a group |
| `H` | `kv_history_chunks` | History window — free (`≥0`) on every topology, independent of both the ordering and the clean step |
| Cell `(c,s,b)` | — | Schedule task + KV version (+ `req` when batched) |
| Tick | — | One cell on one rank (one block forward) |
| `R` | `max_batch_size` | Max micro-batch requests per rank per tick |

### What is a cell?

A **cell** `(c, s, b)` is the schedule/KV atom — one DiT **block** forward on
one tick — not a whole chunk and not a full denoise step.

| | | |
| --- | --- | --- |
| `c` | chunk | Temporal latent slice |
| `s` | step | Denoise index (`0…D-1` noisy; `D` clean when present) |
| `b` | block | Transformer block index |

```text
chunk c → step s → cell (c,s,b)   # one tick / rank = one block
```

History KV is matched on the **same** `b` only. With `K>1` local blocks, one
`(c,s)` spans multiple ticks.

Three orderings; Latest-KV and Same-Step share the diagonal (vertical) topology:

| Strategy | Advance | History KV |
| --- | --- | --- |
| **Self-Forcing** (`Ordering.SELF_FORCING`) | Finish one chunk before the next | Clean `(c', D, b)` only |
| **Latest-KV** (`Ordering.LATEST_KV`) | Diagonal cell overlap | Newest finished `(c', s*, b)` with completion tick strictly earlier |
| **Same-Step** (`Ordering.SAME_STEP`) | Same diagonal placement | Matching step `(c', s, b)`; fixed clean sink from chunk 0 |

`S = 1` is native layer PP. Vertical schedules use `S ∈ {D, D+1}`:
`S = D` (Same-Step; first-chunk sink uses the last replica) or `S = D+1` (one weight replica per noisy step
plus clean). Activations still follow PP (`next_rank` after the block that
feeds the next group). KV is keyed **per block** `b`: publish / transfer /
wait when that cell finishes; a remote reader of `b` does not wait for the
producer's other local blocks.

CausVidWan accepts Self-Forcing / Latest-KV only. Pair schedule with topology: Self-Forcing →
`S=1` layer PP only (`deploy/causvid_wan_self_forcing.yaml`); Latest-KV →
vertical `S=D+1` (`deploy/causvid_wan_latest_kv.yaml`). Clean is an ordering
property, independent of the history window (Self-Forcing and Latest-KV always,
Same-Step runs it for chunk 0 only).

Acceptance for #102: reproducible basic function + recorded config. No FPS /
speedup SLA. DreamZero / LingBot paged-session defaults must not regress.

## Scope

| Axis | Unit |
| --- | --- |
| **Model / weight split** | Unchanged PP: ranks own contiguous layer groups (`G = ⌈B/K⌉`); `pipeline_parallel_size = S·G` |
| **Schedule / KV** | Cell `(chunk, step, block)` — one transformer block per tick |

`cell_schedule` (request `extra_args`) must match the deployment topology:
`S=1` → Self-Forcing, vertical `S=D+1` → Latest-KV (CausVidWan derives the
default from `step_replicas` and rejects a mismatched explicit choice);
Same-Step (`S=D`) is a generic schedule primitive with a fixed first-chunk clean
sink; CausVidWan does not expose it. It is not a session property.
Optional session only keeps clean KV across generations
([Realtime AR-Diffusion sessions](realtime_ar_diffusion.md)).

## Architecture

```text
s = pp_rank // G ,   g = pp_rank % G ,   rank(s, g) = s·G + g
```

Layer split uses `(g, G)` via `get_pp_indices(B, pp_rank % G, G)`, not
`world = S·G`. Production Wan2.2 PP stays on its existing path.

```text
Optional Session
        │
        ▼
ARDiffusionModelRunner
  ├─ Legacy → paged KV (DreamZero / LingBot)
  └─ Cell   → NoisyKV + CellPlan
        │
   cell_schedule │ cell_executor (tick loop, PP P2P)
        └────► kv_cache/noisy.py
```

### How it works

1. The pipeline builds a `CellSchedule` from request `extra_args` and calls
   `build_cell_plan()` (pure function, no torch).
2. `ARDiffusionModelRunner` binds an `ARDiffusionCellContext` over a shared
   `NoisyKVCache` for the request.
3. `run_cell_pipeline()` admits inflight plans, runs one cell per tick on each
   rank, and overlaps KV / activation P2P with compute.
4. Rank 0 gathers finished chunk latents (or streams them) for VAE decode.

## Cell schedule

Implemented in `experimental/ar_diffusion/cell_schedule.py`.

- **In:** `CellSchedule(N, D, S, G, B, K, ordering, H)`.
- **Out:** `CellPlan` — per-tick cells `(c,s,b)`, plus `sources` / `transfers` /
  `last_use` keyed by cell.

Visibility for history of the same block `b`:

```text
# Latest-KV / Self-Forcing (Self-Forcing exposes only clean by then)
s* = max { s | P(c', s, b) < current_tick }
src = (c', s*, b)

# Same-Step: same step × block → same rank → local only
src = (c', s_task, b)   # only if P(c', s_task, b) < current_tick
# no ordinary-history KV transfers under Same-Step; sink is an exception
```

Invariants:

- `S ∈ {1, D, D+1}`
- Same-Step ⇒ `S = D` (no per-chunk clean; `S=1` only when `D=1`); Self-Forcing ⇒ `S = 1` (layer PP); Latest-KV ⇒ `S = D+1` (clean)
- No Latest-KV edge inside the same micro-batch / tick
- Cross-rank KV ready before use (Latest-KV / Self-Forcing when remote)
- Ordinary Same-Step history never needs KV P2P; the clean sink is distributed once
- Finite `H` + `last_use` bounds residency
- Consumer of `b` never waits on producer's `b' ≠ b` (activation edges are separate)

## Cell executor

Implemented in `experimental/ar_diffusion/cell_executor.py`.

Per tick on each rank:

1. Wait transfers required by this cell's `sources` (same `b`).
2. Forward one block via `CellAdapter`; **publish KV immediately**.
3. Post planned KV `isend` (do not wait for remote recv).
4. Wait posted handles; `evict` versions past last-use once sends complete.

Received versions are readable only on a later tick. Same-tick producer /
consumer of the same version is forbidden.

`run_cell_pipeline()` owns admit / drop / fail-closed cleanup so a mid-request
exception does not leak `VersionPool` slots. Rank topology is described by
`CellTopology` / `CellRunSpec` inside `ARDiffusionCellContext`.

Failure cleanup is **request-scoped**: a failed cell releases only the requests
that execution context owns (`ARDiffusionModelRunner._release_cell_requests()`),
not the whole pool. `NoisyKVState.reset_all()` remains reserved for tearing down
the cache itself, because it also drops versions belonging to requests that were
never part of the failing execution.

## NoisyKV

Implemented in `experimental/ar_diffusion/kv_cache/noisy.py`.

| | |
| --- | --- |
| Unit | K/V for `(req, c, s, b)` — one block |
| Storage | Per-layer `VersionPool` from `ARDiffusionNoisyKVSpec` |
| Transport | Planned PP-group P2P; post on cell complete |
| Eviction | `last_use` per cell; no in-place overwrite of live sends |

### Lifecycle contract

The paged session KV and the cell-scoped `VersionPool` keep their own storage and
version semantics. A shared execution flow needs one contract from either
implementation, expressed on the operations rather than on the data layout:

| Question | Contract |
| --- | --- |
| Ownership | Versions belong to the request that produced them; the runner owns the pools. A session may outlive requests but never owns cell versions. |
| Identity | `(request_id, chunk, step, block)` for a cell version; paged sessions keep their own branch/adapter identity. |
| Validity | A version is consumable only after its producing cell completed on an earlier tick and, when remote, after `await_ready`. |
| Completion | A returned `forward` is not proof that K/V writes and transfers are visible; that boundary belongs to the transport completion contract (#9368). |
| Dependency | `CellPlan.sources` / `transfers` / `last_use` decide who still reads a version. |
| Reclamation | `evict` releases a version only after its last planned use and after any send referencing it completed. |
| Failure | Cancellation and exceptions release exactly the failed request's versions, leaving unrelated resident requests intact. |

Per-tick KV traffic ≈ history depth × one block × tokens × heads × dim per
remote source. Mitigations: post-on-produce, dedupe `(version, dst)`, larger `G`.

### Session integration (P1 target contract)

> Status: target contract for the P1 integration. Not implemented yet; the
> current cell path is request-scoped and releases everything it produced.

**Ownership.** A session owns the history that later requests must continue from.
A request owns the versions it produces for its own plan plus any temporary
scratch. Both live in the same `VersionPool`; ownership decides what survives
request termination, not where the bytes sit.

**Chunk numbering.** A request admitted into a session with committed history
starts at `chunk_offset = committed_chunks`. Cell coordinates stay
request-local, while the session records absolute chunk positions, so
`chunk_history_chunks` never reaches back into chunks the session already owns.

**Retention set.** The versions a session keeps after a request are exactly
those a later request can be planned to read:

| Kept | Dropped at request end |
| --- | --- |
| The last committed clean (or Same-Step step) version per block, per retained chunk | Versions whose `last_use` fell inside this request |
| Up to `H` most recent retained chunks plus any sink chunk | In-flight receive buffers and staging |
| Model-declared state the pipeline commits (conditioning, latent, decoder) | Per-request scratch slots |

Retention is bounded by the same `H` / sink policy the plan already uses, so a
session can never keep more history than a single request could consume.

**State machine.** `ACTIVE` admits requests. A successful request commits its
retention set and returns to `ACTIVE`. `reset` discards the retention set and
returns to `ACTIVE` with `chunk_offset = 0`. `close` moves to `CLOSING`, waits
for compute and transfers belonging to the session to finish, releases slots and
model-owned state, then `CLOSED`. Cleanup failure leaves `CLEANUP_FAILED` and
rejects reset, matching the realtime session tombstone.

**Failure policy.** A failed request, an asynchronous device error, or a broken
transport leaves the retention set in an unknown state, so the session becomes
`INVALID`: further requests are rejected until an explicit reset. Late results
from the failed request are fenced by the request identity and must not update
session state.

**Completion boundary.** A version may enter the retention set only after its
producer's completion is established under the transport contract above; a
returned `forward` is not sufficient evidence.

### Same-Step with a first-chunk clean sink

`Ordering.SAME_STEP` uses a first-chunk clean sink for multi-chunk requests.
The topology stays `S=D`. Chunk 0 runs one extra clean forward on the last
replica, then its clean KV is distributed once to the readers before later
chunks start. When the clean successor is on the same rank, the executor carries
the final latent directly into that pass, without adapter-local latent storage.
Subsequent chunks use the normal Same-Step schedule. A single-chunk
request skips the unused clean forward and sink distribution.

Each chunk reads the fixed clean sink plus its most recent `H` Same-Step
history chunks, deduplicating chunk 0 while it is in the window. `H=0` keeps
the sink. This mode currently admits one request at a time; checkpoint quality
requires separate validation.

## Configuration

| Layer | Key | Meaning |
| --- | --- | --- |
| `parallel_config` | `pipeline_parallel_size` | `S·G` |
| `ar_diffusion_stage_config` | `step_replicas` | `S` |
| | `max_batch_size` | `R` |
| | `max_history_chunks` | Cap for `H` |
| Request `extra_args` | `cell_schedule` | CausVidWan: `self_forcing` \| `latest_kv`; Same-Step is schedule-level only |
| | `num_chunks`, `num_denoise_steps`, `kv_history_chunks` | `N`, `D`, `H` (clean scheduling follows the ordering and multi-chunk shape) |

Backend: `ARDiffusionEngine` + mp executor from deploy YAML.
Last-noisy finish rank: `S=1` → `G-1`; vertical `S ∈ {D, D+1}` → `D·G-1`
(used for `chunk_timeline_file`). Remote `stream_decode` `source_rank` is
that finish rank only when `S > 1`. `S=1` decodes locally: `G=1` (single
rank, which owns the VAE) keeps local `stream_decode`; `G>1` rejects it
(fail fast — the finish rank `G-1` has no VAE).

## Compatibility

Only-`SupportsARDiffusionPipeline` paths keep today's behavior. Default
DreamZero / LingBot YAMLs must not silently enter `run_cell_pipeline`.

| Combination | Status |
| --- | --- |
| No session + Self-Forcing / Latest-KV | Supported (opt-in; CausVid) |
| No session + Same-Step | Generic schedule primitive with a clean sink; not exposed by CausVidWan |
| Session + legacy loop | Supported (unchanged) |
| Session + Cell path | Designed; not required for #102 |

**CausVidWan S=1 default:** if request `extra_args` omits `kv_history_chunks`,
`H` defaults to model `max_history_chunks` (not `0`) — that adds the history
window (and its KV reuse) versus the previous unset→`H=0` behavior; set
`kv_history_chunks: 0` to run without history. The clean step is unaffected:
`self_forcing` always runs it (ordering property).

**S=1 without the clean step is intentionally gone:** `self_forcing` always
runs the clean step regardless of `H`. Same-Step (`S=D`, including `S=1`
when `D=1`) runs clean only for chunk 0. `kv_history_chunks: 0` removes recent
history but retains the Same-Step sink.

Capability: `SupportsARDiffusionCellPipeline` —
`ar_diffusion_noisy_kv_spec()`, `bind_ar_diffusion_cell_context()`.

## Testing

CPU: `tests/diffusion/ar_diffusion/test_cell_{schedule,executor}.py`,
`test_noisy_kv*.py`, and model/config resolve tests under the same tree.

GPU (record config): `S=1` Self-Forcing; `S=D` Same-Step; `S=D+1` Latest-KV
(`benchmarks/` / recipe smoke).

## Open issues

- Paged session vs NoisyKV still separate prealloc; no TP × Noisy KV yet.
- Noisy KV not wired to session; the P1 contract above is the target.
- Micro-batch packing (`R`): varlen vs pad.
- Cell schedule with `K>1` multiplies ticks; activation inbox is keyed by
  `(chunk, step)` (R=1 friendly).

## Acceptance cases (P1: CausVid, CPU-first)

Every case records model, checkpoint, `S`/`G`/`D`/`H`, latent shape, dtype, seed,
revision and command. Structure and lifecycle cases run on CPU; output quality is
compared frame-wise against the recorded baseline, and a case fails on a
regression beyond 2% in chunk cadence, interaction latency or peak memory.

| Case | Level | Passes when |
| --- | --- | --- |
| `test_cell_executor`, `test_noisy_kv*`, `test_cell_schedule` | CPU | unchanged green |
| Request-scoped failure release | CPU | a failed request frees only its own versions |
| Session: two consecutive requests | CPU | the second starts at `chunk_offset = committed_chunks` |
| Session: two interleaved sessions | CPU | versions and chunk numbering stay isolated |
| Session: reset mid-sequence | CPU | no read of pre-reset history; offset returns to 0 |
| Session: close then request | CPU | rejected, no silent recreation |
| Session: failed request | CPU | session `INVALID`; further requests rejected until reset |
| Cancel between pieces | CPU | owned versions released; unrelated resident requests intact |
| Self-Forcing `S=1` smoke | 2 GPU | finite latents, chunk count and order correct |
| Latest-KV `S=D+1` smoke | 4 GPU | same, plus a cross-rank history read |
| Baseline parity | GPU | frame-wise match; no metric beyond the 2% budget |

GPU rows stay recorded as unvalidated until a slot is available. A CPU pass must
not be reported as distributed-execution or output-quality evidence.

## Related

- [Realtime AR-Diffusion sessions](realtime_ar_diffusion.md)
- [AR-Diffusion pipeline capability](../ar_diffusion_pipeline_capability.md)
- [Pipeline Parallel](pipeline_parallel.md)
- Recipe: [`recipes/Physis-AI/CausVid-Wan.md`](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Physis-AI/CausVid-Wan.md)
- Code: `experimental/ar_diffusion/{cell_schedule,cell_executor,kv_cache/noisy}.py`
