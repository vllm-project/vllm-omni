# AR-Diffusion KV backends and compute members

The AR runner uses the existing Request and Stepwise entry points. Noisy PP
remains inside the Pipeline's Request forward; no cell-level Engine mode,
wave protocol, or implicit paged-to-versioned conversion is introduced.

## Backend boundary

`KVBackend[TaskT, ViewT]` defines `prepare(tasks)` and `publish(tasks)`.
Preparation lends one ordered read/write view to each logical computation;
publication confirms successful model writes. Models choose which products
become history. Publication does not mean device or transfer completion.

| Backend | Logical unit | Task | Model-facing view |
| --- | --- | --- | --- |
| Legacy chunk/window | One chunk's computation | `PagedKVTask(branch, seq_len, commit_current)` | Paged layer contexts, lowering to `ARDiffusionPagedLayerInputs` |
| Noisy versioned | One Cell/block computation | Owner and `(chunk, step, block)` | Noisy layer contexts, lowering to the same attention inputs |

`ARDiffusionKVState` and `NoisyKVState` implement the backend contract.
Existing legacy `get_kv_caches` and `commit_paged_context` enter through
`prepare` and `publish`; the Cell executor uses the same operations. Tasks
retain their backend-specific coordinates: a Cell is not made into a token
range, and a legacy chunk is not forced to produce a version at every step.
The existing exact-token segmented interface remains separate; it is not
silently converted into a paged attention view.

Primary pools are allocated at model initialization. Runtime allocation and
release transfer ownership of pages or slots inside those pools; they do not
create and destroy the backing GPU tensors on every logical operation.
Backend-specific staging growth and model-private conditioning remain explicit.

## History lifecycle

`KVCacheLifecycle[OwnerT]` distinguishes request computation completion from
history lifetime:

- `finish_request(owner)` retires temporary invocation state, retaining the
  model's selected history.
- `release_history(owner)` returns retained storage after computation has
  retired. Session reset/close, eviction and fail-closed cleanup use this
  boundary. Legacy owners may be branch adapters; the session-facing state
  and Noisy backend also accept the service Session namespace.

Noisy retention and active-owner accounting belong to the shared cache, not
a temporary invocation bridge. A new request can bind retained history through
a new matching plan. Two active invocations cannot own the same namespace.
A plan's last-use table is necessary but not sufficient for reclamation:
computation must have completed, communication must no longer use the storage,
no planned consumer may need it, and history policy must no longer retain it.
Failed transfer waits leave storage and pending work intact for cleanup retry.

Clean-history policies retain clean outputs. Same-Step retains the window's
step-specific versions plus the absolute chunk-zero clean sink. A persistent
single-chunk bootstrap produces the sink for later requests; an ephemeral
single-chunk request preserves its no-sink execution. Initialization budgets
account for retained versions separately from history chunk count.

Service Session lifecycle is reused, not reimplemented here. `kv_request_binding`
resolves typed ticks before legacy extra arguments, providing the same owner,
reset/close flags and generation position to Runner and Pipeline.

## Compute plans and batches

`ComputePlan` provides bounded rank-local task lookup for both `CellPlan` and
`ChunkStepPlan`. Backend-specific dependency generation and communication stay
with the corresponding plan/executor. The chunk executor advances through the
penultimate denoise step and yields before the interaction boundary. The runner
keeps request preparation and Session binding rather than that advancement loop.

`ExecutionBatch` owns ordered `BatchMember` values: request identity, computation
position, inputs, KV view and result-row count. It checks unique members and
maps exact result counts/rows back to their owners before state publication.
The mainline Stepwise core and Cell executor use this common boundary without
replacing `InputBatch` tensor preparation or model numerical updates.

CausVid groups compatible requests from one admitted Request batch and runs
one Cell loop for each compatible group. Each request owns conditioning,
sampler, seed, latent progress, KV views and outputs. Activation packets carry
member coordinates. Equal token counts alone do not establish compatibility:
activation shape, schedule structure and topology also have to match.
Same-Step and remote stream-decode groups remain serialized where their
current protocols cannot safely multiplex members. Output order matches input
request order. Failure visits all created decode jobs and touched owners,
without discarding unrelated history.

This member dispatch is not a claim of fused multi-member model kernels.
Within a Cell batch, the current Wan adapter still computes members separately.
Neither Request batching nor the common member interface supplies dynamic
in-forward admission or Request intermediate video delivery.

## Scope and verification

CPU regressions cover both real storage pools, retention through a fresh
invocation, publication ownership, invalid/stale tasks, computation/consumer/
history reclamation gates, transfer-wait retries, chunk boundaries, member row
mapping, CausVid batch equivalence and failure isolation, and StagePool affinity.
CPU tests do not validate distributed NCCL ordering, real model quality,
throughput or long-running stability.

Scheduler-owned reservations, per-rank grants and same-Session service queuing
are deferred. Existing prefix/CFG/connector ownership is unchanged. Distributed
model validation and matched quality/performance acceptance remain required
before advertising those configurations as validated.
