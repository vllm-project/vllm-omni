# Deploy YAML (Stage Config)

Schema: `DeployConfig` and `StageDeployConfig` in `vllm_omni/config/stage_config.py`.
User docs: `docs/configuration/stage_configs.md`,
`docs/configuration/gpu_memory_utilization.md`. The source is authoritative when
docs disagree.

## Contents
- Selecting a config
- Top-level keys
- Per-stage keys
- Layering: base_config, platforms, CLI overrides
- Common topologies
- Rules of thumb

## Selecting a config

| Mechanism | How |
|-----------|-----|
| Default | Pipeline's `default_deploy_config_name` → `vllm_omni/deploy/<name>.yaml` |
| Explicit | `--deploy-config path.yaml` (offline: `Omni(..., deploy_config=...)`) |
| Per-stage tweak | `--stage-overrides '{"1": {"max_num_seqs": 32}}'` |
| Toggle chunking | `--async-chunk` / `--no-async-chunk` |

`--stage-configs-path` and the old `stage_args` / `engine_args` layout are
rejected. Do not write new configs in that shape.

## Top-level keys

| Key | Default | Meaning |
|-----|---------|---------|
| `base_config` | – | Parent YAML to overlay |
| `pipeline` | – | Override the pipeline registry key (e.g. a thinker-only variant) |
| `async_chunk` | resolved from pipeline | Streaming hand-off between stages |
| `session_mode` | `turn` | `turn` (HTTP request/response) or `duplex` |
| `model_runner` | `v1` | `v2` enables MRv2 runners (`vllm_omni/worker_v2/`) |
| `connectors` | – | Named connector specs: `{name: ConnectorClass, extra: {...}}` |
| `stages` | `[]` | Per-stage settings, matched by `stage_id` |
| `platforms` | – | `cuda` / `rocm` / `npu` / `xpu` / `musa` → nested `stages:` overrides |
| Pipeline-wide engine args | – | `trust_remote_code`, `dtype`, `quantization`, `enable_prefix_caching`, … applied to every stage |
| `duplex_session` | defaults | e.g. `max_sessions`, `idle_ttl_s` |

## Per-stage keys

| Key | Meaning |
|-----|---------|
| `stage_id` | Must match a pipeline stage |
| `devices` | Visible devices for the stage, e.g. `"0"`, `"1,2"`, `"0-3"` |
| `num_replicas` | Replica count; device count must equal replicas × per-replica world size |
| `gpu_memory_utilization` | Fraction of **total** GPU memory for this stage |
| `max_num_seqs`, `max_num_batched_tokens`, `max_model_len` | Usual vLLM limits |
| `enforce_eager` | Disable CUDA graphs for this stage |
| `tensor_parallel_size` | TP within the stage |
| `model_runner` | Per-stage v1/v2 override |
| `async_chunk` | `false` opts a stage out |
| `input_connectors` / `output_connectors` | Connector names |
| `default_sampling_params` | Stage sampling defaults |
| `env` | Per-stage environment variables |
| Other keys | Passed through as engine extras (model-specific knobs) |

`model_stage`, `execution_type`, `worker_type`, and stage input processors are
pipeline fields, not deploy fields.

## Layering

Effective config = parent (`base_config`) → this file → `platforms.<current>` →
CLI (`--stage-overrides`, `--async-chunk`). When debugging, reconstruct the
effective value for the current platform before blaming a knob.

## Common topologies

```yaml
# Thinker alone on GPU 0; talker + code2wav share GPU 1
stages:
  - stage_id: 0
    devices: "0"
    gpu_memory_utilization: 0.9
  - stage_id: 1
    devices: "1"
    gpu_memory_utilization: 0.6
  - stage_id: 2
    devices: "1"
    gpu_memory_utilization: 0.1
```

```yaml
# Scale the talker with replicas
base_config: minicpmo_4_5.yaml
stages:
  - stage_id: 1
    devices: "1,2"
    num_replicas: 2
```

Single-GPU co-location is possible (see `qwen3_omni_moe_mrv2_h200.yaml`) when the
utilization values sum to at most 1.0.

## Rules of thumb

- Stages sharing a GPU: sum of `gpu_memory_utilization` ≤ 1.0.
- Put the thinker on its own GPU first; talker and code2wav are smaller and
  usually share.
- When the talker saturates under concurrency, add talker replicas before raising
  thinker batch limits.
- Prefix caching is off in the full Qwen3-Omni and MiniCPM-o deploys; keep it off
  unless the pipeline variant explicitly supports it.
- Prefer a new YAML with `base_config` over copying a whole file.
- `tools/configure_stage_memory.py` still parses the legacy `stage_args` layout;
  verify it against the current schema before relying on it.
