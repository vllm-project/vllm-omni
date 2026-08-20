# Pipeline and deploy configurations

In vLLM-Omni, a model's `PipelineConfig` defines its fixed stage topology, while a deploy configuration controls how those stages run.

!!! note
    Default deploy config YAMLs (for example, `vllm_omni/deploy/qwen2_5_omni.yaml`, `vllm_omni/deploy/qwen3_omni_moe.yaml`, and `vllm_omni/deploy/qwen3_tts.yaml`) are bundled and loaded automatically when `--deploy-config` is omitted. The resolved pipeline selects its default through `default_deploy_config_name`.

## Pipeline configuration

`PipelineConfig` and its `StagePipelineConfig` entries are Python definitions
owned by the model implementation and registered in `OMNI_PIPELINES`. They
describe fixed model topology and are not accepted in deploy YAMLs.

Common `PipelineConfig` fields include:

| Field | Description |
|-------|-------------|
| `model_type` | Pipeline identifier used during model and config resolution. |
| `default_deploy_config_name` | Bundled deploy YAML loaded when the user does not pass `deploy_config`. |
| `model_arch` | Default Hugging Face architecture for the pipeline. |
| `hf_architectures` | Architecture names used to identify checkpoints whose `model_type` is shared. |
| `hf_config_predicate` | Optional predicate used to select between pipelines with otherwise identical HF metadata. |
| `diffusers_class_name` | Diffusers `_class_name` used to identify Diffusers-style repositories. |
| `stages` | Ordered tuple of fixed `StagePipelineConfig` definitions. |

Common `StagePipelineConfig` fields include:

| Field | Description |
|-------|-------------|
| `stage_id` | Stable stage identifier. |
| `model_stage` | Logical stage name used by runtime and strategy resolution. |
| `execution_type` | `LLM_AR`, `LLM_GENERATION`, or `DIFFUSION`. |
| `input_sources` | Upstream stage IDs that provide this stage's inputs. |
| `final_output` / `final_output_type` | Whether the stage produces a user-visible output and its modality. |
| `owns_tokenizer` | Whether this stage owns the pipeline tokenizer. |
| `model_arch`, `hf_config_name` | Stage-specific model architecture and nested HF config selector. |
| `engine_output_type` | Runtime output representation such as `text`, `latent`, or `audio`. |
| `custom_process_input_func` | Processor applied to this stage's incoming payload. |
| `custom_process_next_stage_input_func` | Processor used for full-payload handoff to the next stage. |
| `async_chunk_process_next_stage_input_func` | Processor used for async chunk handoff. |
| `sampling_constraints` | Model-owned sampling constraints that deploy defaults cannot override. |

To add or change topology, define and register a new pipeline variant. Use
deploy YAML only for runtime placement, resource sizing, connectors, and other
deployment overrides.

## Deploy configuration schema

The new deploy schema lives under `vllm_omni/deploy/` and is paired with a frozen `PipelineConfig` registered by the model's `pipeline.py`. Each deploy YAML has these top-level fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `base_config` | str (path) | optional | — | Overlay parent (relative or absolute). `stages:` / `platforms:` deep-merged by stage_id; other scalars overlay-wins. Intended for user-authored overlays; prod yamls stay flat. |
| `async_chunk` | bool | optional | `true` | Enable chunked streaming between stages. Pin to `false` if the pipeline runs end-to-end. |
| `session_mode` | str | optional | `"turn"` | Session behavior. Use `"duplex"` only with a pipeline that enables duplex control. |
| `active_stream_window` | int | optional | `0` | Number of active downstream stream slots; `0` preserves all-stream cycling. |
| `duplex_session` | dict | optional | runtime defaults | Full-duplex session lifecycle, buffering, replay, and capacity limits. |
| `connectors` | dict | optional | `null` | Named connector specs (`{name, extra}`). Referenced by each stage's `input_connectors` / `output_connectors`. See [Connector schema](#connector-schema). |
| `edges` | list | optional | `null` | Explicit edge list for the KV transfer graph. Auto-derived from stage inputs if omitted. |
| `stages` | list | optional | `[]` | Per-stage runtime overrides matched by `stage_id`. Pipeline stages are still created from `PipelineConfig` when this list is empty. |
| `platforms` | dict | optional | `null` | Keyed by `npu` / `rocm` / `xpu`, each contains a `stages:` list with per-platform overrides applied on top of the CUDA defaults. |
| `pipeline` | str | optional | `null` | Override the auto-detected pipeline registry key (used for structural variants like `qwen2_5_omni_thinker_only`). |
| `trust_remote_code` | bool \| null | optional | `null` | **Pipeline-wide.** Trust HF remote code on model load; applies to every stage when specified. |
| `distributed_executor_backend` | str \| null | optional | `null` | **Pipeline-wide.** Distributed executor backend forwarded to vLLM (`"mp"`, `"ray"`, `"external_launcher"`). If omitted, vLLM auto-selects backend from runtime topology. |
| `dtype` | str \| null | optional | `null` | **Pipeline-wide.** Model dtype for every stage. |
| `quantization` | str \| null | optional | `null` | **Pipeline-wide.** Quantization method for every stage. |
| `enable_prefix_caching` | bool \| null | optional | `null` | **Pipeline-wide.** Prefix cache toggle applied to every stage when specified. |
| `enable_chunked_prefill` | bool \| null | optional | `null` | **Pipeline-wide.** Chunked prefill toggle applied to every stage. |
| `data_parallel_size` | int \| null | optional | `null` | **Pipeline-wide.** DP degree for every stage. |
| `pipeline_parallel_size` | int \| null | optional | `null` | **Pipeline-wide.** PP degree for every stage. |
| `custom_voice_dir` | str \| null | optional | `null` | **Pipeline-wide.** Directory containing custom voice profiles for supported TTS models. |

For fields whose deploy default is `null`, the deploy layer contributes no
override. The effective value may still come from a platform section, an
explicit CLI or stage override, or the downstream vLLM engine default.

Note: for diffusion path, `distributed_executor_backend` currently defaults to
`mp`, and `ray` / `external_launcher` are not fully supported yet.

### Stage fields

Each entry under `stages:` accepts any `StageDeployConfig` field directly (no nested `engine_args:`). Only fields whose value legitimately varies across stages live here; pipeline-wide settings (trust_remote_code, distributed_executor_backend, dtype, quantization, prefix/chunked prefill, DP/PP sizes) are declared at the top level and applied to every stage. Unknown keys fall through to `engine_extras:` and are forwarded to the engine. Frequently used fields are listed below; the source-of-truth schema is `StageDeployConfig` in `vllm_omni/config/stage_config.py`.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `stage_id` | int | required | — | Stage identity; matched against `PipelineConfig.stages[*].stage_id`. |
| `max_num_seqs` | int \| null | optional | `null` | Max concurrent sequences per stage. |
| `gpu_memory_utilization` | float \| null | optional | `null` | Per-stage total memory target; used for automatic KV-cache sizing. |
| `kv_cache_memory_bytes` | int \| null | optional | `null` | Explicit per-rank KV-cache budget in bytes; overrides automatic sizing. |
| `tensor_parallel_size` | int \| null | optional | `null` | TP degree for this stage. |
| `enforce_eager` | bool \| null | optional | `null` | Disable CUDA graphs. |
| `max_num_batched_tokens` | int \| null | optional | `null` | Per-stage prefill/token budget; also contributes to the native maximum in-flight token limit. |
| `max_model_len` | int \| null | optional | `null` | Per-sequence context or KV length; `-1` enables native cache-capacity auto-fitting, while values above the HF default auto-set `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. |
| `async_scheduling` | bool \| null | optional | `null` | Per-stage async scheduling toggle. |
| `devices` | str \| null | optional | `null` | Device list assigned to this stage. |
| `output_connectors` | dict \| null | optional | `null` | Keyed by `to_stage_<n>`; values are names registered under top-level `connectors:`. |
| `input_connectors` | dict \| null | optional | `null` | Keyed by `from_stage_<n>`; values are names registered under top-level `connectors:`. |
| `default_sampling_params` | dict \| null | optional | `null` | Baseline sampling params. Deep-merged with pipeline `sampling_constraints` (pipeline wins). |
| `engine_extras` | dict | optional | `{}` | Catch-all for engine fields not listed above; deep-merged across overlays and forwarded to the stage engine. |

### Connector schema

Each entry under top-level `connectors:` follows this shape:

```yaml
connectors:
  <connector_name>:
    name: <ConnectorClassName>     # required — class registered in vllm_omni.distributed
    extra:                         # optional — forwarded to the connector's __init__
      <key>: <value>
      # Additional connector-specific options
```

| Connector class | Use case | `extra` keys |
|-----------------|----------|--------------|
| `SharedMemoryConnector` | Same-host KV transfer between stages (default for bundled YAMLs). | None. All payloads use shared memory. |
| `MooncakeStoreConnector` | Cross-host KV transfer over TCP. Required for multi-node deployments. | `host`, `metadata_server`, `master`, `segment` (int bytes), `localbuf` (int bytes), `proto` (`"tcp"` / `"rdma"`). |

A stage references a connector by name in its `input_connectors` / `output_connectors`:

```yaml
connectors:
  shm:
    name: SharedMemoryConnector

stages:
  - stage_id: 0
    output_connectors: {to_stage_1: shm}
  - stage_id: 1
    input_connectors:  {from_stage_0: shm}
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--deploy-config PATH` | Load a deploy YAML. **Optional** — when omitted, the bundled `vllm_omni/deploy/<model_type>.yaml` is auto-loaded by the model registry. |
| `--stage-overrides JSON` | Per-stage JSON overrides, e.g. `'{"0":{"gpu_memory_utilization":0.5}}'`. Per-stage values always win over global flags. |
| `--async-chunk` / `--no-async-chunk` | Flip the deploy YAML's `async_chunk:` bool. Unset (default) leaves the YAML value in force. |

### Stage-Based CLI Paradigm

The stage-based CLI paradigm facilitates the execution of discrete pipeline stages within isolated processes:

- **Stage 0** typically encapsulates the orchestrator and the primary API server. Invocation requires `--stage-id 0`,
  `--omni-master-address`, `--omni-master-port`, and standard port declarations (e.g., `--port`).
- **Worker Stages** operate without a distinct API server (i.e., using `--headless`), are assigned sequential `--stage-id` identifiers, and must reference the corresponding
  `--omni-master-address` and `--omni-master-port` parameters to successfully register with Stage 0.

For migrated architectures, the system automatically resolves and loads the bundled deployment YAML. Consequently, the primary execution path
does **not** necessitate the explicit definition of `--deploy-config`:
the example below uses `CUDA_VISIBLE_DEVICES=0` for Stage 0 and
`CUDA_VISIBLE_DEVICES=1` for Stage 1.

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --port 8091 \
    --stage-id 0 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 1 \
    --headless \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

When instantiating a custom deployment YAML, append the `--deploy-config /path/to/override.yaml` directive to all node invocations.

In the context of standard initialization architectures, utilizing the `--stage-overrides` parameter operates as the optimal methodology
for delineating stage-specific tuning from the CLI interface:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --stage-overrides '{"1": {"gpu_memory_utilization": 0.5}}'
```

Conversely, in the context of the **stage-based CLI** paradigm, given that each execution process exclusively instantiates a single pipeline stage, configuration override attributes
can be defined uniformly via explicit CLI flags on the corresponding instantiation command, rendering composite `--stage-overrides` JSON strings unnecessary:

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 1 \
    --headless \
    --gpu-memory-utilization 0.5 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

### Precedence

From highest to lowest:

1. Per-stage overrides (`--stage-overrides` JSON)
2. Explicit global CLI flags (`--gpu-memory-utilization 0.85`, etc.)
3. Platform section (`platforms.npu.stages`, etc.) on top of the base `stages:`
4. Overlay YAML (via `base_config:`) on top of the base YAML
5. Parser defaults

### Worked override example

Starting from the bundled `vllm_omni/deploy/qwen3_omni_moe.yaml`:

```yaml
# vllm_omni/deploy/qwen3_omni_moe.yaml (excerpt)
async_chunk: true
stages:
  - stage_id: 0
    gpu_memory_utilization: 0.9
    max_num_seqs: 32
  - stage_id: 1
    gpu_memory_utilization: 0.7
    max_num_seqs: 16
```

A user-authored overlay that inherits the base and overrides only stage 1:

```yaml
# my_overrides.yaml
base_config: /path/to/vllm_omni/deploy/qwen3_omni_moe.yaml
stages:
  - stage_id: 1
    gpu_memory_utilization: 0.5     # smaller GPU
```

Launched with both an explicit global flag and a per-stage override:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --deploy-config my_overrides.yaml \
    --max-model-len 16384 \
    --stage-overrides '{"0": {"max_num_seqs": 8}}'
```

Within the stage-based CLI paradigm, equivalent configuration parameters can inherently be passed directly
as command-line arguments to the designated single-stage process instantiation:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 0 \
    --max-num-seqs 8 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

Effective config per stage after the merge:

| Stage | Field | Final value | Source |
|-------|-------|-------------|--------|
| 0 | `gpu_memory_utilization` | `0.9` | base YAML (overlay didn't touch stage 0) |
| 0 | `max_num_seqs` | `8` | per-stage CLI (`--stage-overrides`) — wins over base `32` |
| 0 | `max_model_len` | `16384` | global CLI |
| 1 | `gpu_memory_utilization` | `0.5` | overlay YAML — wins over base `0.7` |
| 1 | `max_num_seqs` | `16` | base YAML (overlay didn't touch this field) |
| 1 | `max_model_len` | `16384` | global CLI |
| 2 | (all defaults) | — | base YAML (no overrides apply) |

Therefore, as a core part of vLLM-Omni, a model's pipeline and deployment configurations have several main functions:

- Claim partition of stages and their corresponding class implementation in `model_executor/models`.
- The disaggregated configuration for each stage and the communication topology among them.
- Engine arguments for each engine within the stage.
- Input and output dependencies for each stage.
- Default input parameters.

To override specific parameters, explicitly inject the customized configuration schema
in both online and offline instantiation flows. Use the `--deploy-config` flag
when loading a deploy configuration.

Examples:

For offline inference (assuming the necessary dependencies have been imported):
```python
model_name = "Qwen/Qwen2.5-Omni-7B"
omni = Omni(model=model_name, deploy_config="/path/to/deploy_config.yaml")
```

For online serving:
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 --deploy-config /path/to/deploy_config.yaml
```

!!! important
    We are actively iterating on the definition of deployment configurations, and we welcome feedback from users and developers.
