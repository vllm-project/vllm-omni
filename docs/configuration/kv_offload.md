# KV Cache Offload (Thinker AR Stage)

When the thinker AR stage runs out of GPU KV blocks it must either preempt
requests and re-prefill them later, or push KV blocks somewhere else. This
recipe covers the second option: spilling KV (and hidden states) to a CPU
cache so a request can resume without recomputing prefill.

The established backend is **LMCache** via vLLM's `LMCacheConnectorV1`.
An opt-in Mooncake KV/HS backend for Qwen3-Omni MRv2 turn serving is described
below. Hidden states are
stored alongside KV because downstream stages (talker, code2wav) read the
full thinker HS sequence — restoring KV alone would leave a hole for the
cached prefix.

!!! note
    vLLM v1's `OffloadingConnector` is **not** wired up on this path. An
    earlier iteration used it but it was removed (per-step CPU overhead was
    too high), so all CPU spill flows through LMCache.

## Requirements

- A single GPU is sufficient.
- `pip install lmcache` and (optionally) a reachable LMCache server or
  config file.

## Configuration surface

All options live on the **per-stage** `omni_kv_config` field:

```yaml
omni_kv_config:
  kv_store_config:
    lmcache_config:
      config_file: ""               # "" -> LMCache defaults
                                    # path -> exported as LMCACHE_CONFIG_FILE
      # Any other key here is forwarded to LMCache as lmcache.<key>.
      # Example: max_local_cpu_size: 4.0
```

`lmcache.enable_hidden_state_cache` is forced on automatically whenever
`lmcache_config` is present, so the `HiddenStateStore` is created without
a manual flag.

## Usage

Override the thinker stage in the
[default Qwen2.5-Omni deploy config](gh-file:vllm_omni/deploy/qwen2_5_omni.yaml).
Add the LMCache block and disable in-GPU prefix caching on the same stage
(LMCache manages prefix-keyed CPU storage of its own):

```yaml
stages:
  - stage_id: 0
    enable_prefix_caching: false
    omni_kv_config:
      kv_store_config:
        lmcache_config:
          config_file: ""    # or /path/to/lmcache.yaml
```

```bash
vllm serve Qwen/Qwen2.5-Omni-3B --omni --port 8091 \
    --deploy-config /path/to/qwen2_5_omni_kv_offload.yaml
```

With this configuration the thinker stores per-layer hidden states into
LMCache alongside KV, keyed by the same chunk hashes. A later request that
hits the cached KV prefix will also recover the matching HS, so the talker
receives a complete HS sequence even though prefill was skipped for the
cached tokens.

If LMCache is not installed, the engine refuses to start (early, with a
clear error). Set `config_file` to a YAML when you need to point at a
remote LMCache instance or override defaults; an empty string uses
in-process defaults, which is the right pick for local testing.

## Python API

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="Qwen/Qwen2.5-Omni-3B",
    deploy_config="qwen2_5_omni_kv_offload.yaml",
)
outputs = omni.generate(prompts, omni.default_sampling_params_list)
omni.close()
```

The end-to-end test
[tests/engine/test_kv_offload_with_model.py](gh-file:tests/engine/test_kv_offload_with_model.py)
is the canonical executable example.

## Operational notes

- `enable_prefix_caching` must be `false` on the stage that uses LMCache. The
  engine refuses the combination at kv-cache init: KV received through a
  connector is reported as `num_computed_tokens` too, and the omni prefix cache
  cannot tell it from a local hit. Prefix reuse still happens — it is served
  from LMCache rather than from GPU memory.
- Make sure `LMCACHE_CONFIG_FILE` (set automatically from
  `lmcache_config.config_file`) or the default config points at a reachable
  backend. Misconfiguration surfaces at engine init, not at first request.
- Partial HS retrieves can occur when LMCache's HS pool is smaller than the
  KV pool and the HS LRU evicts faster, which the shipped defaults already
  make likely (2 GB against 5 GB). The runner detects this per layer, logs an
  error, and restores nothing. The request still degrades: its KV hit has
  already skipped the prefill, so the talker is conditioned on the suffix
  alone. Sizing the HS pool at least as large as the KV pool makes this less
  likely, but it is not a correctness guarantee — the pools still evict
  independently. The guarantee needs the scheduler to cap the hit at what both
  tiers cover.
- This recipe targets the AR thinker stage. The diffusion / talker stages
  do not consume `omni_kv_config`.

## Mooncake KV and hidden-state caching (experimental)

The Mooncake backend stores prompt-prefix KV together with the captured input
embeddings and selected-layer hidden states required by Qwen3-Omni Talker.
It wraps vLLM's Mooncake Store **KV connector**. It is independent of Omni's
similarly named inter-stage `MooncakeStoreConnector`, which transports stage
outputs and does not add prefix reuse.

This is the M1 implementation of [RFC #9908](https://github.com/vllm-project/vllm-omni/issues/9908).
M0 validated waiting and recovery with a sequential Thinker-only probe. Those
results do not establish full text/audio serving correctness or performance
for this backend. Full-pipeline GPU acceptance, duplex evaluation, and
performance measurements must be recorded separately before claiming support.

The adapter targets vLLM `0.33.0` at source revision
`4ba787ec9959e566d3b587a796dff07673324da5` and Mooncake `0.3.12.post1`.
Its metadata and worker lookup integration must be rechecked before upgrading
vLLM. CPU regression tests and native host-memory Store tests do not validate
GPU transfers or the complete serving pipeline.

### Configuration

Use an explicit Qwen3-Omni MRv2 turn deployment with CUDA and Thinker TP=1.
Keep the deployment's normal async-chunk handoff and downstream stage settings.
Set this on the Thinker stage:

```yaml
stages:
  - stage_id: 0
    model_runner: v2
    enable_prefix_caching: false
    omni_kv_config:
      kv_store_config:
        mooncake_config:
          config_file: /path/to/mooncake-store.json
          namespace: qwen3-omni-turn-v1
          model_revision: 26291f793822fb6be9555850f06dfe95f2d7e695
          max_hs_bytes: 268435456
          max_pending_transfers: 16
          load_timeout_seconds: 30.0
```

`config_file` selects the upstream Mooncake Store JSON connection configuration.
Use a reachable Mooncake master and install the Mooncake package matching the
selected vLLM/CUDA environment. `namespace` separates cache populations;
`model_revision` must identify the exact checkpoint actually loaded, including
when loading a local directory. Change the namespace for an incompatible cache
deployment. Never reuse a checkpoint identifier for different weights.

`max_hs_bytes` bounds temporary HS data, independently of Mooncake's configured
store capacity. `max_pending_transfers` bounds queued HS transfers, and
`load_timeout_seconds` sets the HS load deadline. Account for GPU KV, temporary
worker buffers, and Mooncake KV/HS/metadata storage separately.
The output tensors handed to Talker are accounted separately from transfer
staging. A timed-out HS read can drain in the background while its request
recomputes, retaining its staging reservation until it finishes. Native KV
copies must finish before their destination GPU blocks can be recycled;
the HS deadline cannot forcibly interrupt a native KV copy.

The configuration bridge selects `OmniMooncakeStoreConnector` through vLLM's
connector extension mechanism and explicitly sets
`kv_load_failure_policy="recompute"`. Upstream's default is `"fail"`, which
terminates the request; it does not provide recomputation. Selecting Mooncake
together with LMCache, a separate `kv_transfer_config`, local Thinker prefix
caching, or a producer-only/consumer-only role is rejected. Keep Talker prefix
caching disabled when comparing Thinker cache behavior.

### Correctness contract

- A lookup identifies a candidate prefix; it is not proof that loading worked.
  Both KV and HS must finish loading and validate before Thinker resumes.
- Restored prompt rows are combined with fresh rows in token order before
  Talker input construction. Each request attempt consumes the restored prefix
  once, so subsequent output chunks cannot prepend it again.
- A failed load clears incomplete restored state and bypasses external lookup
  for that request's recovery. Other requests still use the cache normally.
- Cache writes include only complete prompt blocks. Failed or skipped writes
  must not change the current response or publish an incomplete reusable entry.
- Cancellation and preemption invalidate the load attempt. Buffer memory cannot
  be released while a transfer still accesses it, and a late completion must
  never publish output for a cancelled attempt.

Validation must cover first writes, confirmed full/partial hits, block-boundary
and chunked-prefill cases, corrupted/missing objects, failure recovery, and
request cleanup. Confirm the restored row positions and resulting Talker inputs
as well as text/audio output; matching text alone cannot validate HS restoration.

The opt-in full-pipeline acceptance test requires two H100-class GPUs and a
running Store. It runs two cache-off baselines before the cached arm, records
the actual restore count, and compares the prompt tensors before/after Talker
conversion, generated token IDs, and audio. It uses a fresh cache namespace
and includes chunked prefill, a complete hit, and a shared-prefix partial hit.
The test enables `VLLM_BATCH_INVARIANT=1` for exact comparisons across the
different prefill shapes produced by cache hits. All stages use greedy
sampling and serial request capacity, retaining the deployment's async
handoff and CUDA graphs:

```bash
MOONCAKE_CONFIG_PATH=/path/to/mooncake-store.json \
VLLM_BATCH_INVARIANT=1 \
  python -m pytest tests/engine/test_mooncake_turn_consistency_qwen3.py \
    -v --run-level advanced_model
```

Set `MOONCAKE_TEST_MODEL` to a local copy of the pinned Qwen checkpoint when
needed. Tensor and restore observations are retained in pytest's temporary
directory. The test requires exact baseline reproducibility; if that fails,
establish baseline-derived tolerances before evaluating cached results. The
test's file writes and save barriers are correctness instrumentation and must
not be used for performance measurements.
