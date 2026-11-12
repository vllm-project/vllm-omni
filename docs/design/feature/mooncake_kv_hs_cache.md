# Mooncake KV and Hidden-State Cache

This page is the implementation contract for the experimental Mooncake KV/HS
backend, milestone M1 of
[RFC #9908](https://github.com/vllm-project/vllm-omni/issues/9908).
Configuration, supported deployments, and validation commands are in the
[KV offload guide](../../configuration/kv_offload.md#mooncake-kv-and-hidden-state-caching-experimental).

## Scope

M1 serves ordinary Qwen3-Omni turn requests on CUDA with Model Runner V2: an
unquantized Thinker with tensor parallel size 1, one full-attention KV group,
and equal KV and hash block sizes. The connector rejects other deployments at
startup, including runtime quantization, whose effective model values are not
represented by the checkpoint configuration alone.

**Cache mode.** M1 is a Mooncake-only external cache. Selecting this backend
rejects local Thinker prefix caching, so every reused prefix is restored from
Mooncake. It is not a two-level cache; the
[deferred two-level design](#deferred-two-level-cache-cooperation) records what
that mode requires. Duplex sessions and performance evaluation are M2 work.
Online connector reset is unsupported.

In this page, **HS** means the two Thinker captures that Talker reads for
prompt positions: the input embeddings and the hidden states of
`talker_config.accept_hidden_layer`.

## Ownership

| Component | Process | Owns |
| --- | --- | --- |
| Native `MooncakeStoreConnector` (vLLM `4ba787ec`) | Scheduler and worker | KV keys, KV loads and saves, GPU block references held by saves |
| `OmniMooncakeStoreConnector`, scheduler role | Engine core | Lookup deadlines, recovery bypass, load-attempt tokens, cancellation notices |
| `JointStoreWorker` | Worker rank 0 | Joint KV/HS lookup over the native lookup channel |
| `OmniMooncakeStoreConnector`, worker role | Worker | HS reads and writes, staging budget, restored rows, receive completion |
| `vllm_omni/worker_v2/mooncake.py` | Worker | Step snapshots and the Thinker capture rewrite before output publication |

## Stored data and identity

- **Namespace digest.** SHA-256 of the configured `namespace` and
  `model_revision`, the stage, format version, capture layer, model
  configuration, HS dtype, and KV cache spec. The model configuration excludes
  `_name_or_path` and `transformers_version`, so replicas that load the same
  revision from different paths share entries. The digest is the native KV
  `cache_prefix` and part of every HS key.
- **KV objects.** Native per-block KV keys under that prefix.
- **HS objects.** `omni-hs-v1:<namespace digest>:<prefix digest>`, one
  safetensors object per complete prompt block. Its metadata records the
  namespace, prefix hash, token range, capture layer, hidden size, and dtype.
  The prefix digest comes from the KV connector's chained block hash, which
  includes multimodal input identity.
- **Completion record.** `<HS key>:ready` holds the SHA-256 digest of the HS
  object. It is written only after both the HS object and the block's KV object
  exist.

A record is never trusted alone. A lookup requires the KV object, HS object,
and record for each block. A load checks the digest, then the full header
against the expected layout before tensors are allocated.

## Request lifecycle

### Lookup

The scheduler queries Mooncake asynchronously (native `lookup_async`): a
request whose lookup is in flight stays waiting while other requests are
scheduled. Worker rank 0 answers with the longest contiguous prefix whose KV
object, HS object, and completion record all exist, in whole blocks, capped so
that the read fits `max_hs_bytes`.

A lookup that exceeds `load_timeout_seconds` is treated as a miss. That request
computes its whole prompt without another lookup. The native client
sends lookups one at a time over one socket, so later requests are also misses,
without a lookup, until the stalled reply returns. The connector detects that
reply by queueing a marker job behind the pending lookups, which is valid only
while the client runs lookups on a single worker in submission order; startup
fails if the pinned client no longer does.

### Load

When the scheduler admits a hit, the native connector copies KV into the
allocated GPU blocks. The Omni connector reads the HS objects and completion
records on dedicated read workers, in Store calls of up to 16 MiB. Receive
completion is reported only after both finish and validate; the restored rows
then belong to that request attempt.

A read that finds a missing or invalid object, misses `load_timeout_seconds`,
or is not admitted reports a failed receive. vLLM's `recompute` policy then
recomputes the prefix without another lookup.

### Row assembly

The first output step of an attempt prepends restored rows for the positions
before its first computed token. Each prompt row reaches Talker once, in order:
the restored prefix is consumed once, a replayed final prompt token replaces
its restored row, and a delivered-row watermark drops rows that an earlier,
preempted attempt already handed to Talker.

### Save

Complete prompt blocks are copied out of each step's captures. Saving is best
effort: blocks are skipped when the staging budget or save queue is full, and
the response does not change. A lookup stops at the first missing block, so a
request stops saving after it skips one. Queued saves for one request attempt
run in order; a failed write or expired KV wait also stops later saves while
preserving any successfully saved leading blocks. A save worker writes each HS object
and reads back the stored winner, because a put to an existing key keeps the
first value. It then releases the staging reservation and polls for the
blocks' KV objects until `load_timeout_seconds`. Completion records are
published in block order, stopping at the first block whose KV object is still
missing. If a KV object never appears, no record is written for it or for
later blocks, and normal Store eviction removes the orphaned HS objects.

### Cancellation and preemption

Each scheduled request carries a load-attempt token, so a late result is never
applied to a newer attempt or a reused request ID. Abort and preemption cancel
the attempt: its restored rows are discarded and its pending writes stop. A
normally finished request lets its queued saves finish.

For a cancelled or failed load, receive completion is reported once the native
KV copy has drained, so the scheduler can free the destination GPU blocks. The
CPU reservation is held until the HS read that uses it ends.

### Reset and shutdown

`reset_cache()` returns `False`. The native reset drains native KV writes and
then removes every object in the Store, including other namespaces, while Omni
HS writes may still be running. The failure reaches the caller:
`reset_prefix_cache(reset_connector=True)` returns `False` from the scheduler
and from `AsyncOmni`, and `POST /reset_prefix_cache?reset_external=true`
reports `"success": false`. Entries
written under earlier weights stay in the Store, so a weight change requires a
new namespace or `model_revision`. Shutdown cancels attempts and pending saves
and waits for the HS workers before closing the native Store handle.

## Concurrency and resource limits

| Resource | Bound |
| --- | --- |
| Lookups | Native client: one worker, one request at a time |
| HS reads | Two dedicated workers; at most `max_pending_transfers` admitted reads |
| HS writes | Two dedicated workers; at most `max_pending_transfers` queued saves |
| Temporary HS memory | `max_hs_bytes` covers staging reservations and restored rows |

A read reserves three times its row bytes until it ends. A save releases its
reservation before it waits for KV objects. Because reads and writes use
separate workers and admission slots, a save waiting for a missing KV object
can delay other saves but never a load.

## Failure handling

| Failure | Detected by | Result |
| --- | --- | --- |
| Lookup exceeds `load_timeout_seconds` | Scheduler deadline | Miss without retry; later requests miss until the stalled reply returns |
| Store error during lookup | Worker lookup | Miss |
| KV load error | Native connector | Failed receive, recompute, lookup bypass |
| HS missing, truncated, mismatched, or late | HS read and validation | Failed receive after KV drains, recompute, lookup bypass |
| HS read not admitted | Slot or memory budget | Same as a failed read |
| HS write or read-back failure | Save worker | No completion record; response unchanged |
| KV object never written for a saved block | Save deadline | No completion record |
| Invalid fresh capture | Runner rewrite | Error for that request only |

## Deferred two-level cache cooperation

M1 does not combine Mooncake with local Thinker prefix caching. Combining the
two is outside M1 and needs the follow-up below.

**Current limitation.** Local prefix hits and external loads both advance
`num_computed_tokens`. A local hit finds its Thinker rows in
`PrefixBlockPool`, the [local stage-output cache](prefix_caching.md) indexed by
KV block slot. External loads do not fill that pool, and the runner cannot tell
which source produced a computed token. vLLM also caches externally loaded
blocks locally when prefix caching is enabled, so a later local hit on such a
block would find no Thinker rows.

**Intended design.**

- **Lookup order.** Local GPU prefix first. Mooncake is queried only beyond the
  block-aligned local hit; vLLM already passes that boundary to
  `get_num_new_matched_tokens`, and the native load spec carries both
  boundaries.
- **Hit accounting.** Report local, externally restored, and recomputed tokens
  separately, per request and in aggregate. A failed external load counts its
  tokens as recomputed.
- **KV/HS ownership and row assembly.** For a local hit of `L` tokens and an
  external hit of `E` tokens, rows `[0, L)` come from the local pool, rows
  `[L, L + E)` from Mooncake (read from block `L / block_size`), and later rows
  from computation. The runner must receive `L` and `E` for the attempt rather
  than infer them from `num_computed_tokens`.
- **Promotion and writes.** A block loaded from Mooncake may serve a later
  local hit only after its HS rows are written to the local pool at the same
  block slot; otherwise it must not be cached locally. Mooncake writes stay
  limited to blocks computed by the request.
- **Eviction.** GPU eviction drops a local block and its pool rows together.
  Mooncake evicts independently; readers keep validating, so a missing
  external object is a miss or a failed load.
- **Memory budgets.** Report GPU KV, the local pool (which scales with the GPU
  block count), Mooncake staging (`max_hs_bytes`), and Store usage separately.
- **Failure fallback.** A failed external load discards only the external
  rows. M1 reports failures per request, so vLLM recomputes from token 0 and
  the retry may hit the local cache again; block-level reporting would keep
  the local prefix instead. Local rows must match whichever range remains.
- **Cancellation and reset.** M1 attempt fencing covers the external part;
  local hits have no asynchronous state. A local reset clears GPU blocks and
  the local pool. External reset stays unsupported until it can fence HS
  writes and clear only its own namespace.

**Follow-up acceptance criteria.**

1. Restored embeddings, hidden states, packed Talker inputs, generated tokens,
   and audio match cache-off baselines (exactly, or within tolerances derived
   from baseline repeats) for local-only, external-only, and mixed hits, and
   for a later local hit on a block promoted from Mooncake.
2. A failed external read in a mixed hit leaves local rows intact and
   produces the same outputs as the cache-off baseline.
3. Per-request accounting distinguishes local, external, and recomputed tokens.
4. The matched comparison from RFC #9908 section 4: caching off, local prefix
   caching only, and the two-level mode, with the M1 Mooncake-only mode
   reported separately. Workloads cover serial and concurrent requests and
   working sets that fit and exceed the GPU cache and Mooncake capacity.
   Report TTFT, first-audio latency, RTF, throughput, and memory over repeated
   runs.

## Validation

CPU tests cover HS encoding and row assembly (`tests/distributed/test_mooncake_hs.py`),
the connector protocol (`tests/distributed/test_mooncake_connector.py`), the
runner rewrite (`tests/worker_v2/test_mooncake.py`), and configuration
(`tests/engine/test_map_offload_config.py`). The connector tests include a
stalled lookup driven through the production AR scheduler and the pinned
lookup client, saves waiting for KV on real worker threads, completion records
published in block order, cancellation during an HS read, and a rejected reset
reaching the scheduler's caller.

Opt-in tests need a running Mooncake master and `MOONCAKE_CONFIG_PATH`:
`tests/distributed/test_mooncake_hs_store.py` exercises the native Store with KV
sentinels, and `tests/engine/test_mooncake_turn_consistency_qwen3.py` runs the
cold, full-hit, and partial-hit pipeline on GPUs. The acceptance test compares
embeddings, hidden states, packed Talker inputs, generated tokens, and audio
against independent cache-off runs. Image and audio cases also check that
changing media prevents reuse. Neither test runs in CI.

Two manual benchmarks use the same pinned Qwen3-Omni MRv2 deployment:

- `benchmarks/qwen3_omni/mooncake_cache_benchmark.py` compares cache off, local
  prefix caching, and Mooncake-only caching in separate processes. It records
  client TTFT, first audio, end-to-end latency, RTF, throughput, cache hits,
  restored rows, and memory snapshots. Serial and concurrent waves include
  an overflow working set that exceeds the configured GPU KV capacity.
- `benchmarks/qwen3_omni/mooncake_faults.py` delays lookup and native KV-write
  calls against a real Store, checking unaffected requests' progress, latency,
  and generated tokens. Delays are injected at API entry; they do not model
  Store-internal mutex or transport stalls.

Performance runs use explicit greedy sampling and disable residual-codebook
sampling. They contain no source-tensor dumps or save barriers. Output hashes
and memory snapshots are collected between timed waves. Two-level cache
cooperation and duplex sessions remain outside this validation.

See the [controlled M1 measurements and reproduction commands](https://github.com/vllm-project/vllm-omni/blob/main/benchmarks/qwen3_omni/README.md).
