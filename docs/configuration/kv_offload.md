# KV Cache Offload (Thinker AR Stage)

When the thinker AR stage runs out of GPU KV blocks it must either preempt
requests and re-prefill them later, or push KV blocks somewhere else. This
recipe covers the second option: spilling KV (and hidden states) to a CPU
cache so a request can resume without recomputing prefill.

The backend is **LMCache** via vLLM's `LMCacheConnectorV1`. Hidden states are
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

- `enable_prefix_caching` is independent. Leave it `false` on the thinker
  stage that uses LMCache — the LMCache path does not currently coexist
  with the in-GPU prefix cache for the same blocks.
- Make sure `LMCACHE_CONFIG_FILE` (set automatically from
  `lmcache_config.config_file`) or the default config points at a reachable
  backend. Misconfiguration surfaces at engine init, not at first request.
- Partial HS retrieves can occur when LMCache's HS pool is smaller than the
  KV pool and the HS LRU evicts faster. The runner detects this case per
  layer, emits a warning, and skips writing the truncated tensor into the
  in-GPU prefix cache so downstream stages do not read stale rows.
- This recipe targets the AR thinker stage. The diffusion / talker stages
  do not consume `omni_kv_config`.
