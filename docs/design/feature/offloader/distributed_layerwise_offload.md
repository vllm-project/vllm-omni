# Distributed Layerwise Offload

This document describes distributed layerwise offload (DLO) for diffusion
models. DLO keeps only a small number of DiT blocks on the accelerator and
streams the remaining blocks from host memory. The distributed backend can
either shard those host-side weights across an existing parallel group or keep
complete rank-local block sources and avoid an additional collective.

For user-facing commands, see the
[distributed layerwise offloading guide](../../../user_guide/diffusion/offloader/distributed_layerwise_offload.md)
and the [Cosmos3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-DistOffload.md).

## Feature compatibility

Host-storage optimization and runtime compatibility are separate decisions.
When direct checkpoint mmap is unavailable, DLO can still use tensors produced
by the ordinary loader. "Compatibility path" below means that fallback is
implemented but has less end-to-end coverage than the primary path.

Legend: ✅ supported, ⚠️ compatibility path or limited validation, ❌ unsupported.

| Feature | DLO + AllGather | DLO without AllGather |
| --- | --- | --- |
| **DP** | ✅ Primary path; host weights are sharded across the DP group. | ✅ Each DP rank streams complete rank-local blocks. |
| **SP** | ✅ When DP=1, DLO uses the SP group for weight sharding. | ✅ SP remains active without a DLO weight collective. |
| **TP > 1** | ⚠️ Ordinary TP-aware loader only; no direct checkpoint mmap. | ⚠️ Ordinary TP-aware loader only; no direct checkpoint mmap. |
| **HSDP** | ❌ Rejected to avoid double-sharding parameters. | ⚠️ Limited end-to-end coverage. |
| **Per-tensor online FP8 linears** | ✅ Ordinary loader finalizes weights and scales before DLO sharding. | ✅ Ordinary loader retains complete rank-local tensors. |
| **Online INT8 linears** | ✅ Ordinary loader finalizes the int8 weight (contiguous or transposed view) and fp32 scale before DLO sharding. | ✅ Ordinary loader retains complete rank-local tensors. |
| **Online MXFP8 linears** | ✅ Ordinary loader finalizes the fp8 weight (contiguous) and e8m0 block scale (contiguous or transposed view) before DLO sharding. | ✅ Ordinary loader retains complete rank-local tensors. |
| **Other online quantization methods** | ❌ Rejected until runtime packing and scale layouts are validated. | ⚠️ Allowed through the ordinary loader; validation is method-specific. |
| **Model-level or standard layerwise CPU offload** | ❌ Disabled because DLO takes priority. | ❌ Disabled because DLO takes priority. |
| **Resident leading layers** | ❌ Rejected. | ✅ Requires eligible resident paths in the model's `OffloadPlan`. |

See [Parallelism compatibility](#parallelism-compatibility) and
[Request and loading constraints](#request-and-loading-constraints) for the
detailed contracts and validation boundaries.

## Status

DLO is implemented for multi-device diffusion execution. The default
AllGather path is the primary path for DP and SP deployments. The
`--dlo-no-use-allgather` path streams complete blocks independently and adds no
DLO weight collective.

Host storage is selected separately from the transfer protocol. The loader can
produce a direct-checkpoint mmap plan for a proven-compatible runtime layout;
otherwise it uses the ordinary loader. Consequently, no-AllGather replicas on
the same node can share immutable checkpoint pages when direct mmap is
selected, while the ordinary-loader fallback still keeps a private runtime
copy per process.

The Phase A shared-mmap support boundary is TP1. TP greater than one is an
ordinary-loader compatibility path: DLO can consume the resulting TP-local
tensors, but those configurations do not use checkpoint mmap and must not be
used to claim shared-mmap host-memory savings.

The compatibility matrix below describes the current implementation. The
unit-level guards are covered, but not every parallelism combination has a
full model-and-hardware end-to-end test.

## Design

### DLO consumes the existing parallel topology

DLO does not create a new DP, TP, or SP topology. It reads the configured
`DiffusionParallelConfig` and attaches offload hooks to the DiT blocks after the
standard distributed groups have been initialized.

The DLO weight-sharding group is selected as follows:

1. Use the existing DP group when `data_parallel_size > 1`.
2. When DP is one and SP is greater than one, use the SP group.
3. Otherwise, run rank-locally without a DLO process group.

TP is deliberately not used as DLO's AllGather group. HSDP has its own
parameter-sharding lifecycle and is not allowed to be sharded a second time by
DLO's AllGather path.

### The loader owns host-weight planning

Before it decides whether ordinary weight materialization can be skipped, the
diffusion loader builds one `HostWeightPlan`. A direct-checkpoint mmap plan is
accepted only when preflight proves all of the following:

- every required DiT parameter and persistent buffer has exactly one source;
- runtime names, checkpoint keys, shapes, and dtypes match;
- the runtime topology is TP1 without HSDP or online quantization; and
- every custom loader operation is represented by a loader-owned checkpoint
  adapter.

The exact plan object is handed to DLO. The backend does not rescan checkpoint
files, repeat the capability decision, or reconstruct names from its block
topology. If preflight fails, the loader materializes weights normally and DLO
consumes those runtime tensors.

The plan owns only dedicated DiT component sources. If a pipeline also exposes
ordinary sources for a text encoder or another non-DiT component, the loader
still consumes those sources and includes their loaded names in its strict
coverage check. Only the source prefixes covered by the plan skip ordinary
materialization. A source that mixes DiT and non-DiT weights fails closed to
the complete ordinary-loader path because it cannot be skipped safely as a
unit.

This boundary keeps checkpoint semantics out of DLO and avoids model-pipeline
flags such as `_supports_mmap_loading` or parameter attributes for mmap-only
transforms. Model-specific direct-layout knowledge, when required, lives in a
checkpoint adapter beside the ordinary loader.

### AllGather path

With the default `dlo_use_allgather=True`, each rank stores approximately
`1 / group_size` of each streamable block in pinned host memory. The next
block's shard is copied to a device buffer and reconstructed with
`all_gather_into_tensor` on a communication stream while the current block is
executing.

```text
Compute:    [Block N]             [Block N+1]          [Block N+2]
H2D:                      [shard N+1]           [shard N+2]
AllGather:                [full N+1]             [full N+2]
Buffers:    [current slot]       [prefetch slot]       [current slot]
```

![DLO double-buffer prefetch pipeline](../../figures/dlo/dlo_pipeline.gif)

The backend uses two shared device buffers, so accelerator weight residency is
bounded by the largest streamed blocks rather than the complete model.

When direct checkpoint mmap is selected, the checkpoint mappings are only the
source used to prepare each rank's persistent shard. They can be closed after
shard preparation. Across the AllGather group, those private shards total
approximately one runtime model copy.

An effective DLO group size of one performs no collective, even when
`dlo_use_allgather=True`; it follows the rank-local transfer path described
below.

When DP is greater than one, the engine can process one request per DP rank in
the same denoising wave. Because AllGather is a collective, all participating
requests must take the same execution path at every denoising step.

### Rank-local path without DLO AllGather

With `--dlo-no-use-allgather`, DLO forces its internal offload shard size to
one and streams complete blocks using H2D copies only. The host backing may be
either a loader-approved checkpoint mapping or ordinary runtime tensors.

For direct mmap, each process retains immutable safetensors views and uses two
bounded pinned host staging slots. Processes on the same node that map the same
files share physical checkpoint pages through the OS page cache. This removes
the persistent private full-model copy per pure-DP process, but each process
still packs and transfers every complete block. Sharing is node-local; each
node has its own page cache.

When direct mmap preflight fails, the regular model loader remains responsible
for preparing each rank's weights, including TP-local tensors or HSDP-managed
parameters. In that fallback, each pure-DP process keeps a private full runtime
copy.

This mode means:

- DP still provides independent replicas, but DLO does not shard weights
  across DP ranks.
- SP still performs its normal activation/attention collectives, but DLO does
  not shard weights across SP ranks.
- TP/HSDP/SP collectives, if configured, are not disabled by this flag; only
  DLO's additional weight AllGather is disabled.
- Pure DP deployments share one checkpoint-backed copy per node when direct
  mmap is selected; the ordinary-loader fallback keeps one private runtime
  copy per rank.
- The scheduler does not require a synchronized DP request wave for DLO.

### Component allocator-cache retention

MiniMax-H3 DLO keeps one bounded PyTorch allocator cache across its staged
encoder, DiT, and VAE component boundaries so allocations released by one
component can be reused by the next. The DiT prefetch path and its two shared
device buffers are unchanged.

After a component is offloaded, cached-but-unallocated memory is retained only
while it is at most 25% of device capacity and at least 5% of device capacity
remains physically free. Crossing either bound releases the allocator cache.
Missing allocator telemetry also releases it conservatively. Component or
staging failure forces a release, and an out-of-memory allocation gets one
retry after release. This is a component-local policy rather than a global
`empty_cache` override, so the executor's unconditional shutdown cleanup is
preserved.

Retaining allocator blocks trades higher reserved and peak device memory for
lower lifecycle latency. The benefit is device- and workload-dependent, and a
smaller or externally loaded device may cross a bound and return to the
ordinary per-component cache-release behavior. Current performance validation
covers MiniMax-H3 no-AllGather DLO on CUDA. Other MiniMax-H3 DLO transfer
topologies and platforms receive the same bounded policy but are not claimed
performance targets; other DLO models remain unchanged unless they explicitly
adopt it.

### Final-layout Host Weight Runtime consumer

> **Status:** PR2 landed in
> [PR #6486](https://github.com/vllm-project/vllm-omni/pull/6486). PR3 adds
> registered direct H2D plus generic HWR publication/source-digest
> optimizations without changing AllGather behavior.

Final-layout Host Weight Runtime (HWR) backing is opt-in and currently applies
only to no-AllGather DLO. Enable it with
`--host-weight-runtime-mode preferred` (or `required`) and
`--host-weight-runtime-root <node-local-root>`.

The modes express operator fallback policy, not different artifact formats:

- `preferred` consumes an exact hit, but on a miss it allows canonical loading
  followed by post-load publication. It is the normal population path.
- `required` consumes the same exact artifact but fails startup on a miss or
  unusable artifact. It never invokes canonical DiT fallback or post-load
  publication and therefore cannot populate an empty store.

PR2 has no separate prewarm command. Operators populate one matching producer
cohort per node-local storage domain in `preferred` mode, then restart the same
model revision and parallel layout with `required`. TP coordinates have distinct
identities, while equivalent DP replicas share them.

The current producer/consumer boundary is the model-declared final-layout BF16
contract (MiniMax H3 and `black-forest-labs/FLUX.2-klein-4B`). The FLUX.2-klein
contract covers both transformer block stacks, constructor-stable packed QKV
mapping state, BF16 parameters, and any persistent loader-owned `beta`/`eps`
buffers. The shared HWR machinery supports ordinary-loader final layouts for
TP1 and TP2 rank identities plus SP layout identities. FLUX.2-klein evidence
in this change is TP1-only; its TP2/SP layouts rely on the existing
per-coordinate identity mechanics and remain unmeasured. Online quantization,
HSDP, LoRA/adapted weights, and non-default load formats remain ineligible.
HWR mode
`disabled`, DLO-disabled, and DLO AllGather configurations stop before source
identity or store construction and retain the existing checkpoint-mmap or
ordinary-loader path.

Eligibility is decided before constructing HWR, resolving canonical sources,
hashing identity inputs, probing a filesystem, or emitting an HWR observer
event:

```text
DLO disabled / HWR disabled / DLO AllGather
  -> zero final-layout HWR interaction
  -> preserve current checkpoint-mmap or ordinary-loader behavior
```

When final-layout HWR is explicitly enabled for no-AllGather DLO, runtime
outcomes are authoritative:

- `LOCAL_HIT`: plan and commit an exact restoration, then transfer the lease
  transactionally to DLO.
- `CANONICAL_FALLBACK`: bypass checkpoint mmap, canonically materialize and
  finalize the DiT, then call `publish_after_load()` for a future startup. The
  current startup retains its canonical tensors.
- `FAILED`: fail startup. Preferred mode does not reinterpret nonretryable
  identity, configuration, or compatibility failures as misses.
- Required mode cannot bootstrap an empty store through a `POST_LOAD_ONLY`
  producer.

Checkpoint mmap remains an unchanged control path whenever final-layout HWR is
not selected.

Local source identity and publication avoid repeated full-payload passes while
preserving the same semantic contract:

- Canonical files without a trusted immutable Hub blob identity are hashed in
  parallel. Their content IDs are cached per storage domain under the HWR root
  and reused only when the path, inode, size, timestamps, symlink target, and
  cache-record checksum still match. Corrupt entries are rebuilt; cache-lock or
  cache-I/O failure falls back to hashing the canonical file directly.
- Producers that emit payload bytes in canonical storage-key order may declare
  `ordered=True`. The filesystem writer then computes file and tensor SHA256
  values during the write and overlaps payload `fsync` with later producer
  work. Producers without that guarantee retain unordered writes and use a
  parallel readback checksum fallback.
- Manifest and `READY` publication still wait for every payload `fsync` and
  checksum to complete. These optimizations do not weaken artifact validation
  or change lease ownership.

#### Registered mmap transport

After DLO takes an HWR lease, the no-AllGather backend may register the lease's
complete immutable mapped ranges under the existing `pin_cpu_memory` policy.
Registration is transport state: it does not change artifact identity, the
store, tensor ownership, or H2D payload. On success, each tensor view copies
directly into the existing rotating HBM block buffers and the two private host
staging slots are not allocated.

`--dlo-host-registration-limit-gib` is an optional per-worker preflight ceiling
over page-aligned registered bytes. Zero adds no ceiling. A disabled pinned
memory policy, unsupported platform/capability, over-budget mapping, or fully
rolled-back registration error selects the existing two-slot staging path. A
partial registration that cannot be rolled back aborts startup because closing
the lease would unmap memory still owned by the platform.

Direct checkpoint mmap remains unchanged and continues to use staging. It may
require loader-owned per-block transforms, while the HWR artifact already
contains final runtime bytes. DLO AllGather never receives an HWR final-layout
lease and therefore never enters this registration path.

#### Pre-service transaction boundary

The startup transaction does not end at restore commit. A lease-backed model is
disposable until backend setup and initial prefetch complete:

```text
UNRESOLVED
  -> LEASE_OWNED_BY_LOADER
  -> RESTORE_PLANNED
  -> COMMIT_STARTED
  -> CARRIER_OWNED
  -> BACKEND_OWNED
  -> BACKEND_READY
  -> IN_SERVICE
```

Failure before `COMMIT_STARTED` closes the loader-owned lease. Preferred mode
may canonically load using the untouched skeleton; required mode fails.

Failure from `COMMIT_STARTED` through `BACKEND_READY` must:

1. synchronize or otherwise quiesce partial backend and initial-prefetch work;
2. release hook, staging, restore-plan, and model references;
3. close the lease through its current owner;
4. discard the restored model; and
5. in preferred mode, construct a fresh canonical model while bypassing HWR
   lookup, HWR publication, and checkpoint mmap for that recovery attempt.

Required mode fails instead of constructing the fresh fallback. Once the model
enters service, failures follow normal runtime handling rather than startup
fallback.

#### Finalization phases

Cold canonical loading separates byte-changing work from shared runtime-state
finalization:

```text
Cold-only byte-changing:
  casting, reordering, packing, quantization, generated scales,
  model-specific weight transforms, and calibration that mutates
  parameters or persistent buffers

Shared cold/warm non-byte:
  validation, hook installation, eval state, bookkeeping,
  and non-persistent runtime state
```

The warm path must snapshot restored tensor bytes and backing pointers before
shared finalization and prove both are unchanged afterward.

#### Lease ownership

Lease transfer is single-owner and single-take:

```text
loader owns
  -> restore plan borrows
  -> commit succeeds
  -> runner carrier takes once
  -> backend takes once before asynchronous work
  -> backend drains pending H2D work
  -> backend releases hooks, staging, and model references
  -> backend unregisters every mapped range
  -> backend closes lease
```

The runner carrier is process-local, rejects serialization, and rejects a
duplicate `take()`. If the carrier still owns the lease, runner cleanup closes
it. Once the backend takes ownership, only backend abort or teardown may drain
work and close it.

The implementation keeps the final-layout lease through backend setup and
initial prefetch. Successful registration copies directly from immutable HWR
views; otherwise the backend uses the existing two bounded rank-local staging
slots. Transactional cleanup drains device work, removes source references,
unregisters mappings, and only then closes the lease. An unregistration failure
retains both registration and lease for retry/process teardown. Preferred mode
then uses the runner's fresh canonical retry; required mode propagates the
failure.

#### PR2 and PR3 promotion gates

- Warm hit performs zero ordinary DiT materialization and zero producer calls.
- Shared warm finalization changes neither restored bytes nor backing pointers.
- Preferred `FAILED` outcomes do not silently fall back.
- Planning failure leaves the skeleton untouched.
- Any pre-service failure after commit begins discards the restored model;
  preferred mode uses a fresh canonical model and required mode fails.
- Mixed components load normally and no required tensor remains on `meta`.
- Disabled mode and AllGather emit zero final-layout HWR interaction.
- The lease carrier rejects duplicate take and serialization.
- Backend setup or prefetch failure drains asynchronous work before lease close.
- Registration is all-or-nothing; rollback/unregistration failure never closes
  a mapping still owned by the platform.
- Registered HWR transport bypasses host staging while preserving output and
  H2D payload; unsupported registration retains the bounded staging path.
- Compatible checkpoint mmap remains an unchanged benchmark and control path.
- Prewarm uses one matching producer cohort per TP/SP topology and storage
  domain.
- Benchmarks compare ordinary loading, cold publication, and warm staging with
  output parity, startup latency, aggregate PSS, page-cache state, HBM, and H2D
  payload.

## Parallelism compatibility

| Parallelism | DLO + AllGather | DLO without AllGather |
| --- | --- | --- |
| **DP** | Supported primary path. DLO shards host weights across the DP group and can run DP multi-concurrency. | Supported rank-local path. Compatible TP1 replicas can share checkpoint pages on each node; fallback runtime tensors remain private. |
| **SP** | Supported in the implementation. With DP=1, DLO uses the SP group for host-weight sharding; SP still shards sequence/activation work. | SP remains active, but DLO keeps standard-loader rank-local weights and adds no SP weight collective. |
| **TP > 1** | Outside the Phase A shared-mmap support scope. The loader falls back before mutation, preserves TP-local layouts, and DLO may apply DP/SP host sharding to those ordinary runtime tensors. | Outside the Phase A shared-mmap support scope. The ordinary TP-aware loader produces rank-local tensors, which DLO streams without an additional weight collective; DP replicas retain private runtime storage. |
| **HSDP** | Rejected. HSDP has already sharded parameters, so DLO AllGather would double-shard them. | Accepted by configuration. HSDP owns parameter sharding and its own gathers; DLO only stages rank-local parameters. End-to-end coverage is limited. |

### Combined dimensions

- **DP + SP:** DLO uses the DP group for weight sharding when DP is greater
  than one; SP continues to use its own sequence-parallel group. If DP is one,
  the SP group becomes DLO's sharding group in AllGather mode.
- **DP + TP/SP without AllGather:** standard model loading defines the
  rank-local tensor layout. DLO adds no cross-DP, cross-TP, or cross-SP weight
  collective. When the model declares the final-layout contract, HWR keys the
  reusable artifact by TP rank/size and SP layout.
- **HSDP + SP:** the general parallel configuration permits HSDP over SP, but
  DLO must use `--dlo-no-use-allgather`. HSDP remains responsible for weight
  materialization and synchronization.
- **HSDP + DP or TP:** rejected independently by the diffusion parallel
  configuration.

## Request and loading constraints

AllGather DP multi-concurrency requires:

- explicit `num_inference_steps`;
- the same `num_inference_steps` for all requests in a wave; and
- identical request arguments that affect the collective execution path.

The no-AllGather path does not impose these DLO-specific synchronized-wave
requirements.

Direct checkpoint mmap can back either transfer path. It is currently limited
to proven TP1, non-HSDP, non-online-quantized layouts. Other layouts use the
ordinary loader. Per-tensor online FP8, online INT8, and online MXFP8 linears
can use DLO AllGather after the ordinary loader finalizes their runtime
weights and scales; DLO then shards and reconstructs those tensors with their
recorded layouts (they all keep plain 1-byte dtypes over ordinary strided
views: online INT8 has an int8 weight plus fp32 scale, online MXFP8 has an
fp8 weight plus e8m0 block scale, each with a contiguous or transposed-view
finalized tensor, all covered by the same physical-order packing as online
FP8). Other online methods must use `--dlo-no-use-allgather` or disable
online quantization until their runtime layouts are validated.

The Host Weight Runtime representation, publication, and no-AllGather consumer
contracts are merged; see
[RFC #6414](https://github.com/vllm-project/vllm-omni/issues/6414),
[PR #6445](https://github.com/vllm-project/vllm-omni/pull/6445), and
[PR #6486](https://github.com/vllm-project/vllm-omni/pull/6486). Registered
direct H2D is an optional transport layer over that merged lease contract.

## Validation coverage

Current source-level validation includes:

- HSDP + DLO + AllGather rejection;
- HSDP + DLO without AllGather acceptance at configuration level;
- loader preflight fallback for TP, HSDP, online quantization, unknown custom
  loaders, missing keys, and shape/dtype mismatches;
- ordinary-loader fallback for per-tensor online FP8, online INT8, and online
  MXFP8 linears followed by DLO sharding of finalized weights and scales;
- exact loader-to-backend plan transfer and ordinary-loader fallback;
- ordered publication hashing, overlapped durability, unordered parallel
  checksum fallback, and node-local source-digest reuse/invalidation;
- rank-local mmap source retention, bounded two-slot staging, and adapter
  transforms without parameter-side flags;
- read-only registration preflight, direct source-to-device copies, safe
  fallback, partial rollback, retryable unregistration, and lease ordering;
- resident-layer requests requiring no-AllGather;
- DP request-wave validation for denoising-step compatibility;
- bounded component allocator-cache retention, conservative/forced release,
  OOM retry, and immutable encoder non-block staging;
- sharding, double-buffer, AllGather-size, and heterogeneous-block regression
  tests.

### B300 parallel-topology smoke matrix

A four-GPU B300 smoke test covered MiniMax-H3 FL2VA with the same prompt, seed,
CUDNN attention backend, 256x256 output, two denoising steps, and
`dlo_resident_layers=0`. The TP2 rows used DiT DP2xTP2 with the text encoder and
VAEs at TP1. They validate the ordinary-loader fallback only, not direct mmap
or shared-mmap host-memory savings.

| Configuration | Result | Warm E2E | Peak device memory | Host PSS |
| --- | ---: | ---: | ---: | ---: |
| DP4xTP1 AllGather | Passed, 4 concurrent requests | 2.87 s / 4 requests | 13.84 GiB | 211.99 GiB |
| DP4xTP1 no-AllGather | Passed, 1 request | 15.02 s | 13.23 GiB | 187.77 GiB |
| DP2xTP2 AllGather | Passed, 2 concurrent requests | 4.16 s / 2 requests | 12.50 GiB | 211.97 GiB |
| DP2xTP2 no-AllGather | Passed, 1 request | 3.51 s | 11.88 GiB | 314.01 GiB |

Within each topology, the AllGather and no-AllGather video and audio outputs
were byte-identical. All four runs completed without an `ERROR` or traceback
and released their device allocations. For DP4xTP1, no-AllGather direct mmap
reduced total PSS by 24.22 GiB (11.4%) and `Private_Dirty` from 211.33 to
125.32 GiB (40.7%) relative to AllGather. For DP2xTP2, preflight selected the
ordinary loader as designed; no-AllGather PSS was 314.01 GiB, about 48% above
AllGather, because DP replicas did not share checkpoint-backed runtime
weights. This is a functional and memory smoke test, not a production-quality
performance or output-quality benchmark.

### Host-memory measurement

A two-worker MiniMax-H3 FL2VA measurement on one L20X node compared the
ordinary-loader fallback with direct mmap. Both runs used
DP=2, TP=1, no DLO AllGather, BF16 weights, two denoising steps, and a
256x256 four-second request. The ordinary-loader workers were sampled after
initialization. The mmap workers were sampled after one completed request, so
the checkpoint working set had been faulted into the page cache; this is the
more conservative point for mmap.

The values below come from `/proc/<worker>/smaps_rollup` and include the whole
worker, not only the DiT. The stable rank-to-rank difference comes from other
pipeline components, so each worker should be compared with the same worker in
the other storage mode.

| Worker | Ordinary RSS | mmap RSS | Ordinary PSS | mmap PSS | PSS reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| DP worker 0 | 168.27 GiB | 132.76 GiB | 167.84 GiB | 101.43 GiB | 66.40 GiB |
| DP worker 1 | 116.19 GiB | 79.97 GiB | 115.73 GiB | 48.64 GiB | 67.09 GiB |
| **Two-worker total** | — | — | **283.56 GiB** | **150.08 GiB** | **133.48 GiB (47.1%)** |

The direct-mmap workers each reported 62.45 GiB `Shared_Clean` but only
31.20 GiB `Pss_File`, which is the proportional charge expected when the same
resident checkpoint pages are mapped by two workers. `Private_Dirty` also fell
from 167.53 to 70.24 GiB for worker 0 and from 115.40 to 17.44 GiB for worker
1, a reduction of about 97–98 GiB per worker. RSS understates this benefit
because it counts a shared physical page in every process that maps it; summed
PSS is the appropriate node-memory comparison.

The highest-value missing coverage is broader end-to-end numerical and
lifecycle comparison against ordinary layerwise offload for DP+SP,
HSDP+SP+no-AllGather, and TP greater than one across additional models and
target CUDA/NCCL or CANN/HCCL hardware. That broader TP coverage does not
change the Phase A direct-mmap TP1 support boundary.

## Recommendations

- Use **DP + DLO AllGather** for the supported throughput and host-memory
  scaling path.
- Use **SP + DLO AllGather** for long-sequence workloads when DP concurrency is
  not the goal.
- Use **no-AllGather** when independent replica execution is required. TP1
  direct-mmap deployments can share checkpoint pages per node; other layouts
  retain the ordinary loader's private host memory behavior and are outside
  the Phase A shared-mmap support scope.
- Prefer **HSDP alone** for production HSDP deployments until the combined
  HSDP + DLO no-AllGather path has broader end-to-end coverage.
