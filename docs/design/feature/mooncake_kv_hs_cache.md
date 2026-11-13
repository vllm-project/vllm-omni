# Mooncake KV and Hidden-State Cache

This page is the implementation contract for the experimental Mooncake KV/HS
backend, milestones M1 (turn serving) and M2 (duplex sessions and evaluation)
of [RFC #9908](https://github.com/vllm-project/vllm-omni/issues/9908).
Configuration, supported deployments, and validation commands are in the
[KV offload guide](../../configuration/kv_offload.md#mooncake-kv-and-hidden-state-caching-experimental).

## Scope

The backend serves Qwen3-Omni turn requests and duplex sessions on CUDA with
Model Runner V2: an unquantized Thinker with tensor parallel size 1, one
full-attention KV group, equal KV and hash block sizes, and content-hashed
multimodal inputs. The connector rejects other deployments at startup,
including runtime quantization, whose effective model values are not
represented by the checkpoint configuration alone.

**Duplex sessions.** Qwen3-Omni duplex commits each user utterance as a fresh
Thinker request whose prompt is rebuilt from the selected conversation
history (`Qwen3OmniDuplexPlugin.plan_append`). The connector therefore needs no
duplex-specific state: a follow-up turn reuses the stored blocks of earlier
prompts in the session, an interruption aborts the reply's request, and a
history change that alters earlier tokens or media shortens the hit at the
first changed block. Each committed audio item is re-encoded from the same
PCM, so its content hash, and the block hashes that include it, repeat across
turns.

**Multimodal identity.** Block hashes include each media item's identifier.
vLLM derives that identifier from the item's content unless both the
multimodal processor cache (`mm_processor_cache_gb: 0`) and local prefix
caching are disabled; it then uses a per-process request counter. This
backend always disables local prefix caching, and a counter repeats after a
restart while the Store keeps its objects, so a restarted server could reuse
another request's audio or image prefix. The connector rejects
`mm_processor_cache_gb: 0`.

**Cache mode.** The backend is a Mooncake-only external cache. Selecting it
rejects local Thinker prefix caching, so every reused prefix is restored from
Mooncake. It is not a two-level cache; the
[deferred two-level design](#deferred-two-level-cache-cooperation) records what
that mode requires. Online connector reset is unsupported.

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
that the read fits `max_hs_bytes`. The scheduler then caps the admitted hit at
the request's whole prompt blocks. A preempted request resumes with its
generated tokens, which the lookup also covers, but restored HS covers prompt
rows only; the generated positions are recomputed.

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

A read that finds a missing or invalid object, extends past the prompt, misses
`load_timeout_seconds`, or is not admitted reports a failed receive. vLLM's `recompute` policy then
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
run in order; a successor enters the executor only after its predecessor
finishes, so waiting for that predecessor never occupies another save worker.
A failed write or expired KV wait also stops later saves while
preserving any successfully saved leading blocks. A save worker writes each HS object
and reads back the stored winner, because a put to an existing key keeps the
first value. It then releases the staging reservation and starts a fresh
`load_timeout_seconds` budget for polling the blocks' KV objects. HS encoding
and Store writes do not consume that readiness budget; it is not an overall
save-duration limit and cannot interrupt a blocking Store call. Completion records are
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

## Telemetry

The connector adds its records to the native Mooncake per-operation
telemetry (`MooncakeStoreConnectorStats`), which vLLM aggregates into
`SchedulerStats.kv_connector_stats` and exports as the
`vllm:mooncake_store_operation_*` Prometheus families, labelled by
`operation` and `status`. Each record has a duration, a key count, bytes, and
a failed-key count.

| Operation | Role | One record per | Keys / failed keys | Statuses |
| --- | --- | --- | --- | --- |
| `lookup_wait` | Scheduler | Prefix lookup, as the request waited for it | 1 / 1 if not answered | `ok`, `timeout`, `skipped` (stall guard) |
| `joint_lookup` | Worker rank 0 | Joint lookup | Blocks with KV / those without usable HS | `ok`, `partial_failure`, `error` |
| `joint_load` | Worker | Load attempt, from submission to outcome | Blocks / blocks left to recompute | `ok`, `hs_error`, `kv_error`, `timeout`, `rejected`, `cancelled` |
| `hs_get` | Worker | HS prefix read and validation | HS objects and records | `ok`, `error`, `cancelled` |
| `hs_put` | Worker | Save job's HS object writes | Blocks / blocks not written | `ok`, `partial_failure`, `cancelled`, `skipped` |
| `hs_commit` | Worker | Save job's completion records | Blocks / blocks never published | `ok`, `partial_failure`, `cancelled` |

An `hs_put` record is `skipped` when a full save queue or staging budget turns
blocks away, or when an earlier save of the same request attempt failed and
its queued successors are dropped.

Bytes count HS rows, not encoded objects. `joint_load` bytes are the restored
rows only. Native records cover the KV side: `lookup_exists`, `load_get`,
`save_exists`, and `save_put`. The `hs_commit` duration includes the wait for
the blocks' native KV objects, so it measures how long new blocks take to
become reusable.

The restored token count of a load is its `joint_load` keys times the block
size. vLLM's external prefix-cache counters
(`vllm:external_prefix_cache_queries`, `vllm:external_prefix_cache_hits`) count
lookup hits when the scheduler admits them, before the load succeeds or
fails, so they overstate reuse when loads fail.

**Known gap.** Omni schedulers emit `SchedulerStats` at most once per second
(`OmniSchedulerMixin.make_stats`). A step that emits none discards the
connector records passed to it, so `/metrics` undercounts every Mooncake
operation: native and Omni, worker and scheduler role alike. In a
three-request diagnostic run, all 26 records reached the scheduler and 4 were
exported. Until records from skipped steps are carried into the next emission,
treat these counts as lower bounds.

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

The pinned native connector has one KV send thread. A stalled write can make
unrelated requests' saves miss the deadline even when their HS workers are free.
`load_timeout_seconds` sets the KV-readiness deadline; cache population
remains best effort.

## Deferred two-level cache cooperation

M1 does not combine Mooncake with local Thinker prefix caching. Combining the
two is outside M1 and is tracked in
[feature issue #10335](https://github.com/vllm-project/vllm-omni/issues/10335),
using the design and acceptance criteria below.

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
published in block order, slow HS writes retaining the full KV-readiness budget,
cancellation during an HS read, and a rejected reset
reaching the scheduler's caller.

Opt-in tests need a running Mooncake master and `MOONCAKE_CONFIG_PATH`:
`tests/distributed/test_mooncake_hs_store.py` exercises the native Store with KV
sentinels, `tests/engine/test_mooncake_turn_consistency_qwen3.py` runs the
cold, full-hit, and partial-hit pipeline on GPUs, and
`tests/engine/test_mooncake_duplex_consistency_qwen3.py` runs scripted duplex
sessions on GPUs. The turn acceptance test compares embeddings, hidden states,
packed Talker inputs, generated tokens, and audio against independent cache-off
runs. Image and audio cases also check that changing media prevents reuse.
None of these tests runs in CI.

The duplex test runs the same script on a cache-off server and a Mooncake
server, with audio input and explicit commits: a cold turn, a follow-up that
reuses the session prefix, a reply cancelled while it is generating, a turn
whose HS read is made to fail after the native KV load starts, a turn whose
history leaves the four-audio prompt window, and a second session whose audio
differs only in content. It requires identical prompts, Talker inputs, reply
text, and audio in both servers, and restored row counts equal to the longest
stored block-aligned prefix: zero for the failed read, and ending before the
first audio block for the changed media.
