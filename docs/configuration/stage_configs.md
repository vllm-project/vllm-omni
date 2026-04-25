# Stage configs for vLLM-Omni

In vLLM-Omni, the target model is separated into multiple stages, which are processed by different LLMEngines, DiffusionEngines or other types of engines. Depending on different types of stages, such as Autoregressive (AR) stage or Diffusion transformer (DiT) stage, each can choose corresponding schedulers, model workers to load with the Engines in a plug-in fashion.

!!! note
    Default deploy config YAMLs (for example, `vllm_omni/deploy/qwen2_5_omni.yaml`, `vllm_omni/deploy/qwen3_omni_moe.yaml`, and `vllm_omni/deploy/qwen3_tts.yaml`) are bundled and loaded automatically when neither `--stage-configs-path` nor `--deploy-config` is provided — the model registry resolves the right pipeline + deploy YAML by `model_type`. The bundled defaults have been verified on 1xH100 for Qwen2.5-Omni and 2xH100 for Qwen3-Omni. Models that have not yet migrated to the new schema continue to use the legacy `vllm_omni/model_executor/stage_configs/<model>.yaml` files via `--stage-configs-path`.

## New deploy schema reference

The new deploy schema lives under `vllm_omni/deploy/` and is paired with a frozen `PipelineConfig` registered by the model's `pipeline.py`. Each deploy YAML has these top-level fields:

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `base_config` | str (path) | optional | — | Overlay parent (relative or absolute). `stages:` / `platforms:` deep-merged by stage_id; other scalars overlay-wins. Intended for user-authored overlays; prod yamls stay flat. |
| `async_chunk` | bool | optional | `true` | Enable chunked streaming between stages. Pin to `false` if the pipeline runs end-to-end. |
| `connectors` | dict | optional | `null` | Named connector specs (`{name, extra}`). Referenced by each stage's `input_connectors` / `output_connectors`. See [Connector schema](#connector-schema). |
| `edges` | list | optional | `null` | Explicit edge list for the KV transfer graph. Auto-derived from stage inputs if omitted. |
| `stages` | list | required | — | Per-stage engine args + wiring (see [Stage fields](#stage-fields)). |
| `platforms` | dict | optional | `null` | Keyed by `npu` / `rocm` / `xpu`, each contains a `stages:` list with per-platform overrides applied on top of the CUDA defaults. |
| `pipeline` | str | optional | `null` | Override the auto-detected pipeline registry key (used for structural variants like `qwen2_5_omni_thinker_only`). |
| `trust_remote_code` | bool | optional | `true` | **Pipeline-wide.** Trust HF remote code on model load; applies to every stage. |
| `distributed_executor_backend` | str \| null | optional | `null` | **Pipeline-wide.** Distributed executor backend forwarded to vLLM (`"mp"`, `"ray"`, `"external_launcher"`). If omitted, vLLM auto-selects backend from runtime topology. |
| `dtype` | str \| null | optional | `null` | **Pipeline-wide.** Model dtype for every stage. |
| `quantization` | str \| null | optional | `null` | **Pipeline-wide.** Quantization method for every stage. |
| `enable_prefix_caching` | bool | optional | `false` | **Pipeline-wide.** Prefix cache toggle applied to every stage. |
| `enable_chunked_prefill` | bool \| null | optional | `null` | **Pipeline-wide.** Chunked prefill toggle applied to every stage. |
| `data_parallel_size` | int | optional | `1` | **Pipeline-wide.** DP degree for every stage. |
| `pipeline_parallel_size` | int | optional | `1` | **Pipeline-wide.** PP degree for every stage. |

Note: for diffusion path, `distributed_executor_backend` currently defaults to
`mp`, and `ray` / `external_launcher` are not fully supported yet.

### Stage fields

Each entry under `stages:` accepts any `StageDeployConfig` field directly (no nested `engine_args:`). Only fields whose value legitimately varies across stages live here; pipeline-wide settings (trust_remote_code, distributed_executor_backend, dtype, quantization, prefix/chunked prefill, DP/PP sizes) are declared at the top level and applied to every stage. Unknown keys fall through to `engine_extras:` and are forwarded to the engine.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `stage_id` | int | required | — | Stage identity; matched against `PipelineConfig.stages[*].stage_id`. |
| `max_num_seqs` | int | optional | `64` | Max concurrent sequences per stage. |
| `gpu_memory_utilization` | float | optional | `0.9` | Per-stage memory budget. |
| `tensor_parallel_size` | int | optional | `1` | TP degree for this stage. |
| `enforce_eager` | bool | optional | `false` | Disable CUDA graphs. |
| `max_num_batched_tokens` | int | optional | `32768` | Prefill budget. |
| `max_model_len` | int \| null | optional | `null` | Per-stage context length (auto-sets `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` when larger than HF default). |
| `async_scheduling` | bool \| null | optional | `null` | Per-stage async scheduling toggle. |
| `devices` | str | optional | `"0"` | `CUDA_VISIBLE_DEVICES`-style device list. |
| `output_connectors` | dict \| null | optional | `null` | Keyed by `to_stage_<n>`; values are names registered under top-level `connectors:`. |
| `input_connectors` | dict \| null | optional | `null` | Keyed by `from_stage_<n>`; values are names registered under top-level `connectors:`. |
| `default_sampling_params` | dict \| null | optional | `null` | Baseline sampling params. Deep-merged with pipeline `sampling_constraints` (pipeline wins). |
| `engine_extras` | dict | optional | `{}` | Catch-all for keys not listed above; deep-merged across overlays. Also carries per-stage overrides of pipeline-wide settings (e.g. stage-specific `dtype`). |

### Connector schema

Each entry under top-level `connectors:` follows this shape:

```yaml
connectors:
  <connector_name>:
    name: <ConnectorClassName>     # required — class registered in vllm_omni.distributed
    extra:                         # optional — forwarded to the connector's __init__
      <key>: <value>
      ...
```

| Connector class | Use case | `extra` keys |
|-----------------|----------|--------------|
| `SharedMemoryConnector` | Same-host KV transfer between stages (default for bundled YAMLs). | `shm_threshold_bytes` (int, default `65536`). |
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

### CLI flags introduced in this refactor

| Flag | Description |
|------|-------------|
| `--deploy-config PATH` | Load a new-schema deploy YAML. Takes precedence over `--stage-configs-path`. **Optional** — when omitted, the bundled `vllm_omni/deploy/<model_type>.yaml` is auto-loaded by the model registry. |
| `--stage-overrides JSON` | Per-stage JSON overrides, e.g. `'{"0":{"gpu_memory_utilization":0.5}}'`. Per-stage values always win over global flags. |
| `--async-chunk` / `--no-async-chunk` | Flip the deploy YAML's `async_chunk:` bool. Unset (default) leaves the YAML value in force. |
| `--stage-configs-path` | **Deprecated.** Accepts legacy `stage_args` yamls and (auto-detected) new deploy yamls; emits a deprecation warning. Migrate to `--deploy-config`. To be removed in a follow-up PR. |

### Stage-Based CLI Paradigm

The stage-based CLI paradigm facilitates the execution of discrete pipeline stages within isolated processes:

- **Stage 0** typically encapsulates the orchestrator and the primary API server. Invocation requires `--stage-id 0`,
  `--omni-master-address`, `--omni-master-port`, and standard port declarations (e.g., `--port`).
- **Worker Stages** operate without a distinct API server (i.e., using `--headless`), are assigned sequential `--stage-id` identifiers, and must reference the corresponding
  `--omni-master-address` and `--omni-master-port` parameters to successfully register with Stage 0.

For migrated architectures, the system automatically resolves and loads the bundled deployment YAML. Consequently, the primary execution path
does **not** necessitate the explicit definition of `--deploy-config`:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --port 8091 \
    --stage-id 0 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000

vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 1 \
    --headless \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

When instantiating a custom deployment YAML conforming to the updated schema, append the `--deploy-config /path/to/override.yaml` directive
to all node invocations. For legacy architectures (e.g., BAGEL) configured via deprecated `stage_args:` schemas, continue to specify the relevant configuration via `--stage-configs-path /path/to/config.yaml`.

In the context of standard initialization architectures, utilizing the `--stage-overrides` parameter operates as the optimal methodology
for delineating stage-specific tuning from the CLI interface:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --stage-overrides '{"1": {"gpu_memory_utilization": 0.5}}'
```

Conversely, in the context of the **stage-based CLI** paradigm, given that each execution process exclusively instantiates a single pipeline stage, configuration override attributes
can be defined uniformly via explicit CLI flags on the corresponding instantiation command, rendering composite `--stage-overrides` JSON strings unnecessary:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
    --stage-id 1 \
    --headless \
    --gpu-memory-utilization 0.5 \
    --omni-master-address 127.0.0.1 \
    --omni-master-port 26000
```

### Precedence

From highest to lowest:

1. Per-stage flags (`--stage-overrides` JSON, `--stage-<id>-<key>` if registered)
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
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni \
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

Therefore, as a core part of vLLM-Omni, the stage configs for a model have several main functions:

- Claim partition of stages and their corresponding class implementation in `model_executor/models`.
- The disaggregated configuration for each stage and the communication topology among them.
- Engine arguments for each engine within the stage.
- Input and output dependencies for each stage.
- Default input parameters.

To override specific parameters, explicitly inject the customized configuration schema
in both online and offline instantiation flows. Prioritize the `--deploy-config` flag
when loading the new-schema deploy YAML schemas, reserving the `--stage-configs-path` parameter
exclusively to maintain compatibility with legacy `stage_args` YAML constructs.

Examples:

For offline (Assume necessary dependencies have been imported):
```python
model_name = "Qwen/Qwen2.5-Omni-7B"
omni = Omni(model=model_name, stage_configs_path="/path/to/custom_stage_configs.yaml")
```

For online serving:
```bash
vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 --deploy-config /path/to/deploy_config.yaml
```

Legacy online serving:

```bash
vllm serve ByteDance-Seed/BAGEL-7B-MoT --omni --port 8091 --stage-configs-path /path/to/stage_configs_file
```
!!! important
    We are actively iterating on the definition of stage configs, and we welcome all feedbacks from both community users and developers to help us shape the development!

Below is a specific example of stage_configs.yaml in Qwen2.5-omni.
```python
# stage config for running qwen2.5-omni with AsyncOmniEngine + Orchestrator runtime.
stage_args:
  - stage_id: 0 # mark the unique id for each stage
    runtime: # The disaggregated configuration
      process: true  # Run this stage in a separate process
      devices: "0" # Logical device index for this stage (mapped through CUDA_VISIBLE_DEVICES / ASCEND_RT_VISIBLE_DEVICES if set)
    engine_args: # Engine arguments for a certain engine
      model_stage: thinker
      max_num_seqs: 1
      model_arch: Qwen2_5OmniForConditionalGeneration # The model implementation registered in model_executor/models/registry.py
      worker_type: ar # The specific worker used
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler # The specific scehduler used
      gpu_memory_utilization: 0.8 # The gpu memory allocation for the stage within a single chip
      enforce_eager: true  # Now we only support eager mode
      trust_remote_code: true # Needed by huggingface config parsing
      engine_output_type: latent  # It claims that the stage will input latent hiddenstates besides token ids
      enable_prefix_caching: false # For request with hiddenstates output, the prefix caching is not supported now
    is_comprehension: true # If the stage is a text or multimodal comprehension module. If it is, the AsyncOmni will use its tokenizer as default
    final_output: true # If the stage has output as part of final outputs. If it is false, which means that the stage only works as a intermediate role.
    final_output_type: text # What is the final output type. It can be text and audio now.
    default_sampling_params: # sampling parameters for the stage. Their meaning aligns with vLLM.
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 2048
      seed: 42
      detokenize: True
      repetition_penalty: 1.1
  - stage_id: 1
    runtime:
      process: true
      devices: "1"
    engine_args:
      model_stage: talker
      max_num_seqs: 3
      model_arch: Qwen2_5OmniForConditionalGeneration
      worker_type: ar
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      gpu_memory_utilization: 0.8
      enforce_eager: true
      trust_remote_code: true
      enable_prefix_caching: false
      engine_output_type: latent
    engine_input_source: [0]
    custom_process_input_func: vllm_omni.model_executor.stage_input_processors.qwen2_5_omni.thinker2talker
    default_sampling_params:
      temperature: 0.9
      top_p: 0.8
      top_k: 40
      max_tokens: 2048
      seed: 42
      detokenize: True
      repetition_penalty: 1.05
      stop_token_ids: [8294]
  - stage_id: 2
    runtime:
      process: true
      devices: "0"            # Example: use a different GPU than the previous stage; use "0" if single GPU
    engine_args:
      model_stage: code2wav
      max_num_seqs: 1
      model_arch: Qwen2_5OmniForConditionalGeneration
      worker_type: generation
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      gpu_memory_utilization: 0.15
      enforce_eager: true
      trust_remote_code: true
      enable_prefix_caching: false
      engine_output_type: audio
    engine_input_source: [1]
    final_output: true
    final_output_type: audio
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 2048
      seed: 42
      detokenize: True
      repetition_penalty: 1.1

# Top-level runtime config (concise): default windows and stage edges
runtime:
  enabled: true

  edges:
    - from: 0                   # thinker → talker: trigger only after receiving full input (-1)
      to: 1
    - from: 1                   # talker → code2wav: trigger only after receiving full input (-1)
      to: 2

```

## Stage Configuration Arguments

Each stage in the `stage_args` list contains the following configuration options:

### `stage_id`

A unique identifier for each stage in the multi-stage pipeline. Stages are numbered sequentially starting from 0, and this ID is used to reference stages in inter-stage dependencies (e.g., `engine_input_source`).

### `prompt_expand_func` (Optional)

A custom Python function hook for the LLM stage (Stage 0) that expands a single incoming prompt object into multiple prompts. This is primarily used for multi-modal Classifier-Free Guidance (CFG), where it generates the necessary companion requests (like a negative text prompt) and tags them with internal roles (e.g., `cfg_text`). This ensures the upstream LLM generates the needed contextual hidden states for both the conditional and unconditional generations simultaneously.

### `cfg_kv_collect_func` (Optional)

A custom Python function hook for downstream diffusion stages (Stage 1+) to collect, map, and process the KV caches transferred from the companion requests fired by `prompt_expand_func`. It aggregates the hidden condition states cleanly (e.g., binding them as `cfg_text_past_key_values` and `cfg_text_kv_metadata`), allowing the diffusion runtime to perform CFG smoothly without redundantly evaluating text paths on the DiT workers.

### `runtime`

Configuration for disaggregated execution of the stage, controlling how the stage is deployed and executed.

#### `runtime.process`

Whether to run this stage in a separate process. When set to `true`, the stage will be executed in an isolated process, enabling better resource isolation and parallel execution across different stages. This is essential for multi-GPU deployments where different stages run on different devices.

Default: `true`

#### `runtime.devices`

Logical device indices for this stage, specified as a string. Values are **logical indices** (`0`, `1`, `2`, ...) — not physical GPU IDs — and are mapped through the platform's visibility env var (`CUDA_VISIBLE_DEVICES` on CUDA, `ASCEND_RT_VISIBLE_DEVICES` on NPU) before being applied via `torch.cuda.set_device()` (or the equivalent).

Example: if `CUDA_VISIBLE_DEVICES=0,2,4` is set in the environment, then `devices: "0"` selects physical GPU 0 (the first visible), `devices: "1"` selects physical GPU 2, and `devices: "0,1"` makes physical GPUs 0 and 2 available to the stage. If no visibility env var is set, logical and physical IDs coincide.

Default: `"0"`

#### `engine_args.max_num_seqs`

The maximum number of sequences for concurrent processing in this stage. For LLM stages, this controls the vLLM scheduler's maximum concurrent sequences. For all stage types, this also controls how many tasks can be batched together in the task processing loop.

Default: `1`

### `engine_args`

Engine arguments for configuring the LLM engine, diffusion engine, or other engine types used by this stage.

#### `engine_args.model_stage`

The name identifier for this model stage within the multi-stage architecture. This is used internally to distinguish different stages of the same model (e.g., "thinker", "talker", "code2wav" in Qwen2.5-Omni).

#### `engine_args.model_arch`

The model architecture class name that is registered in `model_executor/models/registry.py`. This specifies which model implementation to use for this stage. The class must be registered in the model registry for vLLM-Omni to locate and instantiate it.

#### `engine_args.worker_cls`

The specific worker class to use for this stage. This determines how the model computations are executed. Examples include `vllm_omni.worker.gpu_ar_worker.GPUARWorker` for autoregressive stages and `vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker` for diffusion-based stages.

#### `engine_args.scheduler_cls`

The scheduler class to use for this stage. The scheduler manages request queuing, batching, and execution order. Examples include `vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler` for standard stages and `vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler` for diffusion stages.

#### `engine_args.gpu_memory_utilization`

The fraction of GPU memory to allocate for this stage within a single GPU chip. This is a value between 0.0 and 1.0, where 0.8 means 80% of the GPU memory will be used by this stage. This allows fine-grained control over memory allocation when multiple stages share the same GPU or when reserving memory for other operations.

Default: `0.8`

!!! tip "Memory Configuration Guide"
    For detailed information on how to calculate memory requirements and properly configure `gpu_memory_utilization`, see the [GPU Memory Calculation and Configuration Guide](./gpu_memory_utilization.md).

#### `engine_args.enforce_eager`

Whether to enforce eager execution mode. When set to `true`, the engine will run in eager mode without using CUDA graphs or other compilation optimizations. Currently, vLLM-Omni only supports eager mode.

Default: `true`

#### `engine_args.trust_remote_code`

Whether to trust remote code when loading models from Hugging Face. This is required for models that use custom code in their configuration files. Set to `true` when loading models that require custom model implementations.

Default: `true`

#### `engine_args.engine_output_type`

Specifies the type of output produced by this stage's engine. This determines what kind of data flows to downstream stages. Possible values include `latent` (hidden states), `text` (tokenized text), and `audio` (audio waveforms). When set to `latent`, the stage outputs latent hidden states in addition to token IDs, which are consumed by downstream stages.

Default: `latent`

#### `engine_args.enable_prefix_caching`

Whether to enable prefix caching for this stage. Prefix caching can improve performance by caching KV cache for common prompt prefixes. However, for requests that output hidden states (when `engine_output_type` is `latent`), prefix caching is not currently supported and should be set to `false`.

Default: `false`

### `is_comprehension`

Whether this stage is a text or multimodal comprehension module. When set to `true`, the stage acts as a comprehension module that processes input text or multimodal content. If this is the first comprehension stage, `AsyncOmni` will use its tokenizer as the default tokenizer for the entire pipeline.

Default: `true`

### `final_output`

Whether this stage produces output that is part of the final outputs returned to the user. When set to `false`, the stage only works as an intermediate stage, processing data that flows to downstream stages but not contributing directly to the final response.

Default: `true`

### `final_output_type`

The type of final output produced by this stage. This specifies what format the output will be in when returned to the user. Currently supported values are `text` (for text generation) and `audio` (for audio generation).

Default: `text`

### `default_sampling_params`

Default sampling parameters for this stage. These parameters control the generation behavior and align with vLLM's sampling parameter semantics. These defaults are used when no explicit sampling parameters are provided in the request.

#### `default_sampling_params.temperature`

Sampling temperature for controlling randomness. Lower values (e.g., 0.0) make the output more deterministic and focused, while higher values increase randomness.

Default: `0.0`

#### `default_sampling_params.top_p`

Nucleus sampling parameter. Only tokens with cumulative probability mass up to `top_p` are considered. This helps filter out low-probability tokens while maintaining diversity.

Default: `1.0`

#### `default_sampling_params.top_k`

Top-k sampling parameter. Only the top `k` most likely tokens are considered. Set to `-1` to disable top-k filtering and consider all tokens.

Default: `-1`

#### `default_sampling_params.max_tokens`

Maximum number of tokens to generate in this stage. This limits the length of the output sequence.

Default: `2048`

#### `default_sampling_params.seed`

Random seed for reproducible generation. When set, the random number generator will be initialized with this seed to ensure consistent outputs across runs.

Default: `42`

#### `default_sampling_params.detokenize`

Whether to detokenize the output tokens into text. When set to `true`, token IDs are converted back to readable text strings.

Default: `True`

#### `default_sampling_params.repetition_penalty`

Penalty applied to tokens that have already appeared in the generated sequence. Values greater than 1.0 discourage repetition, while values less than 1.0 encourage it. A value of 1.0 applies no penalty.

Default: `1.1`

### `tts_args` (TTS stages only)

Configuration for Text-to-Speech specific parameters. This section is only applicable to TTS model stages (e.g., `qwen3_tts`).

#### `tts_args.max_instructions_length`

Maximum character length for voice style/emotion instructions. Instructions exceeding this limit will be rejected with a validation error.

Default: `500`

This value can be overridden at runtime using the `--tts-max-instructions-length` CLI parameter when starting the server.
