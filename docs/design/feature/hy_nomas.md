# Hy-Nomas bounded CPU history archive

> **Status: experimental and opt-in.** C1 stores committed history for later
> sampling and recovery work. It does not enable history restoration, undo,
> pause/resume, or qualify the complete long-video service/quality matrix.

## Scope and source

This implements C1 of [RFC #9525](https://github.com/vllm-project/vllm-omni/issues/9525)
for developers serving controllable, long-running Hy-Nomas video sessions.
It preserves BF16, full history, and eight sampling steps as defaults; CPU
archival is disabled unless explicitly requested for recent-history AR sessions.

The implementation carries forward the C1 delta from
[mooncake2026/vllm-omni#1](https://github.com/mooncake2026/vllm-omni/pull/1),
commit `641943dd`, by `lishunyang12`. The #9338 model integration is already
upstream and is not imported again. C2 and the remaining C/D roadmap items are
separate changes.

The [model integration contract](../module/diffusion/diffusion_model_integration.md)
and [AR session contract](realtime_ar_diffusion.md) own runtime boundaries.
The [recipe](../../../recipes/Tencent/Hy-Nomas.md#experimental-cpu-history-archive-c1)
owns runnable configuration.

## C1: bounded CPU archive

### Ownership and representation

`HyNomasSession.history_archive` owns CPU copies of each newly committed chunk
when `ar_history_mode=recent` and `ar_history_archive_max_bytes > 0`.
The archive contains all completed chunks, including ones still resident on
GPU. A record contains:

- absolute latent start/count and spatial tokens per latent frame;
- segment ID, appended text-token count, and action-text offset;
- per-layer, local-head K/V tensors, already position encoded;
- the original visibility mask, before later segments hide old action text.

Video rows precede text rows. `frame_rows(position)` maps an absolute latent
position to its video slice; text starts at `frame_count * tokens_per_frame`.
Text is appended only when the baseline commits it, including recaching after
recent-history eviction. Transient conditioning rows are excluded.
No GPU segment handle is retained in the CPU archive.

`ARDiffusionSegmentedKVState.pending_layer_views()` borrows initialized current
rows after every layer is staged. The model copies those views before staging
can be overwritten. It does not inspect allocator internals or change the
attention backend, retained set, RoPE positions, or sampling RNG.

### Publication and failure

1. Run the existing required KV-commit forward and stage every layer.
2. Validate the GPU retention set and the archive's prospective byte/frame use.
3. Copy current K/V and mask to CPU, preparing an unpublished record tuple.
4. Use the existing SP error agreement before GPU publication.
5. Commit GPU retention, then publish the prepared archive tuple.

The request-mode runner also releases the session when the base runner returns
an error-bearing `DiffusionOutput`: client errors can be serialized rather than
raised through the session context. The original error/status is preserved;
subsequent use of the same session ID starts with fresh state.

Archive copies are blocking and once per layer/chunk. This adds D2H work;
no speedup is claimed. An archive copy/quota failure participates in the
existing rank error check. Local or peer failure discards pending CPU/GPU
state without publishing it. Reset, close, cancellation, and runner failure
cleanup clear the session archive. This reuses the baseline publication
contract; it does not provide recovery after process/device loss during commit.

### Bounds and limit behavior

`ar_history_archive_max_bytes` limits committed plus pending CPU tensor payload
per session **per rank**, including K, V, and masks. The default `0` disables
archival. `ar_history_archive_max_frames` separately bounds latent positions
and record count; its default is `4096`. Python object/tuple overhead is
bounded by record count but is not included in the tensor-byte metric.

Before copying a chunk, both prospective limits must fit. Otherwise the request
fails explicitly and the runner releases the session. There is no silent
archive truncation or policy downgrade. A deployment should reserve up to
`session_capacity * SP_world_size * archive_max_bytes` host tensor memory,
plus models, Python metadata, decoder/output buffers, and other service state.
The archive quota is separate from the GPU admission ledger and does not
reserve host RAM physically or guarantee that allocation cannot fail.

Chunk metadata reports `archive_payload_bytes` and `archive_latent_frames`
for the local rank; both are zero when archival is disabled. These are payload
counters, not total process RSS or aggregate multi-rank measurements.

An archive is not a resumable checkpoint: RNG, decoder windows, conditioning
objects, interaction queues, and client timeline state are not captured here.
C7/C8 must add and budget those before advertising recovery.

## Validation boundary

CPU tests cover archive payload independence, exact accounting, absolute frame
mapping, quota rejection, missing-layer reads, transient-row exclusion,
peer/local rollback, default-off behavior, session isolation, and reclamation.
The ordinary model and segmented-KV regression suites remain applicable.

Local validation on the earlier upstream base `2b610b3e` compared real BF16
archive-off/on output at 384×640 on SP1/SP2, including repeated history eviction,
copy/quota and rank-local failures, and explicit cleanup checks. A 60-turn SP2
run generated 2,940 frames (122.5 seconds at 24 fps), with identical latents and
pixels for archive off/on. This is a storage-parity result, not a claim of
independent-reference agreement over the complete long run.

The long run contains visual artifacts in both configurations. Archival parity
therefore does not establish long-video quality. The complete public-service
cancellation/backpressure/concurrency matrix, target-resolution qualification,
and W6 remain separate requirements. Recheck against the reported source,
model revisions, controls, and device configuration when promoting a profile.

Report actual host RSS and per-rank archive payload separately. Copy timing
includes synchronization and validation; it is not pure transfer bandwidth or
a speedup claim. CPU tests do not substitute for GPU or service validation.
