# LoRA

vLLM-Omni has two LoRA paths, one per stage execution type:

- **LLM stages** (AR and generation) use vLLM's LoRA. Each stage owns its own
  `LoRAConfig`, and the stage's vLLM engine loads and applies adapters.
- **Diffusion stages** use the diffusion LoRA manager. Its options live on the
  diffusion config, and its usage is described in the
  [Diffusion LoRA guide](../../user_guide/diffusion/lora.md).

The two paths share the `LoRARequest` request type and the `max_cpu_loras`
option name, but no config object and no adapter cache. This page describes
how LoRA options reach each stage and which stage a request-level adapter is
applied to. User-facing configuration is in
[Stage configs](../../configuration/stage_configs.md#vllm-lora-on-llm-stages).

## Supported scope and boundaries

| | LLM stages (vLLM LoRA) | Diffusion stages (diffusion LoRA manager) |
| --- | --- | --- |
| Which models | Runtime `enable_lora` checks upstream `SupportsLoRA` on the loaded stage class when the engine starts (for example BAGEL stage 0 and the HunyuanImage-3 AR stage). Wrapper classes such as `Qwen2_5OmniForConditionalGeneration` and `Qwen3OmniMoeForConditionalGeneration` do not implement `SupportsLoRA` for runtime `enable_lora`, though their inner thinkers do; merged mode (`omni_lora_ipc`) bypasses runtime `enable_lora` and uses `SupportsMergedLoRA` / `packed_modules_mapping` instead. | Pipelines listed in the [Diffusion LoRA guide](../../user_guide/diffusion/lora.md) |
| Enabled by | `enable_lora` on that stage (stage-scoped key in multi-stage pipelines) | `lora_path` / `lora_backend` on the diffusion config |
| Adapters per request | One, on the entry stage only | One or a supported composition (PEFT backend); the distill backend fuses its weights once at initialization |
| Limits | vLLM's `LoRAConfig` options of that stage (`max_loras`, `max_lora_rank`, `max_cpu_loras`, target modules) | `max_cpu_loras` of the diffusion config |
| Runtime adapter management | `AsyncOmni.add_lora` / `remove_lora` / `list_loras` / `pin_lora`, sent to every stage | The same calls; the diffusion worker implements `add_lora`, `remove_lora`, `list_loras` and `pin_lora` |

## LLM stages: one `LoRAConfig` per stage

`OmniStageLoRAConfig` inherits `vllm.config.LoRAConfig` and is a field of the
structured stage config (`BaseVllmOmniStageConfig.lora_config`). It follows the
same reuse rules as the other inherited concern configs; see
[vLLM-Omni config](../module/vllm_omni_config.md#current-structured-stage-reuse-behavior).

- **Built only when enabled.** As with `VllmConfig.lora_config`, a stage
  carries `lora_config` only when its resolved options set `enable_lora`.
  Other LoRA options on a stage without `enable_lora` have no effect.
- **Validated at construction.** The upstream `LoRAConfig` validation runs when
  the stage config is built. `OmniStageLoRAConfig` adds one check,
  `max_cpu_loras >= max_loras` when `max_cpu_loras` is positive, and leaves
  platform-dependent checks to the engine process.
- **Projected back to engine args.** When the stage engine args are built,
  a stage with `lora_config` gets `enable_lora=True` plus the `LoRAConfig`
  fields that were set explicitly, through the generated projection map. The one name difference,
  `target_modules` to `lora_target_modules`, is in that map. A stage without
  `lora_config` gets the same engine args as before LoRA was typed.

### Where options may be set

vLLM LoRA is configured per LLM engine, so the scope of an option matters:

| Pipeline | Global option (`--enable-lora`, `Omni(enable_lora=True)`) | Stage-scoped option (deploy YAML stage entry, `--stage-overrides`, `stage_N_*`) |
| --- | --- | --- |
| Single stage | Applies to that stage | Applies to that stage |
| Multi-stage | Rejected, with an error that lists the LLM stage ids | Applies only to that stage |

Multi-stage pipelines reject global vLLM LoRA options because broadcasting them
would also enable LoRA on downstream LLM stages such as a talker or codec
stage, while a request's adapter is only applied to the entry stage (next
section). The headless / stage-based CLI follows the same rule, since every
process resolves the whole pipeline.

`max_cpu_loras` is the one name both paths use. A global `max_cpu_loras` on a
pipeline that has a diffusion stage keeps its diffusion meaning: it sizes only
the diffusion LoRA cache and is not rejected. An LLM stage's value must then be
set with a stage-scoped key.

Diffusion stages do not accept `enable_lora` and select no vLLM LoRA fields.

## Request-level adapters

A `LoRARequest` passed to `AsyncOmni.generate(..., lora_request=...)` is
attached to the request that the orchestrator builds for the entry stage
(stage 0 for most pipelines). Requests that the orchestrator builds for
downstream stages do not carry it. The entry stage must therefore be an LLM
stage with `enable_lora`, and the adapter must fit that stage's `LoRAConfig`
(rank, number of adapters, target modules).

Forwarding `lora_request` through the offline `Omni.generate` to an LLM entry
stage is added by [#8196](https://github.com/vllm-project/vllm-omni/pull/8196).

`AsyncOmni.add_lora`, `remove_lora`, `list_loras` and `pin_lora` send the call
to every stage through `collective_rpc` and combine the results of the stages
that implement it.

## Diffusion stages

Diffusion LoRA options are fields of `OmniDiffusionConfig`: `lora_path`,
`lora_scale`, `lora_backend` (`peft` or `distill`) and `max_cpu_loras`. The
PEFT backend caches adapters and activates the request's selected adapters;
the distill backend fuses weights once at initialization. See
the [Diffusion LoRA guide](../../user_guide/diffusion/lora.md) for formats,
supported pipelines and checkpoint name mapping
([#8001](https://github.com/vllm-project/vllm-omni/issues/8001)).

Runtime PEFT adapters may be composed into one rank-sum adapter. Before
composition, the manager calls an optional pipeline
`_validate_diffusion_lora_composition(lora_model=...)` hook for each source,
immediately after loading or cache lookup. A pipeline may reject a source
before another transient load prunes its metadata or any weights are composed.
MiniMax-H3 uses this hook to reject FlashGen native stacks; those adapters also
reject merged application during metadata loading because their sampling
contract depends on runtime activation. See
[adapter-declared schedules](../../user_guide/diffusion/lora.md#minimax-h3-adapter-declared-schedules).

## Merged LoRA mode

Engine-side merged LoRA updates live dense weights in place (`W = W_base + (B @ A) * scale` in fp32 from a host-resident pristine bank) without installing runtime LoRA wrappers into the forward module tree.

### Application modes by stage type

| Mode | Stage types | Forward path | Adapter updates |
| --- | --- | --- | --- |
| Runtime wrappers (`lora_application="runtime"`, `enable_lora=True`) | LLM and diffusion | Punica / wrapped linear layers | Per-request `LoRARequest`, `add_lora` / `remove_lora` / `pin_lora` |
| Merged global policy (`lora_application="merged"`, `backend="omni_lora_ipc"`) | LLM (`enable_lora=False`) and diffusion | Unwrapped dense linear layers | Four-phase weight-transfer session while inference is paused |
| Distill init-fuse (`lora_backend="distill"`) | Diffusion | Fused at checkpoint load | Static for process lifetime |

### Serving and lifecycle rules

- A merged engine serves one global policy. Generation requests must omit `LoRARequest`; passing a per-request adapter raises on diffusion stages and runtime `add_lora` / `pin_lora` raise on both stage types.
- `remove_lora` restores the pristine base weights from the host bank and clears the registered merged adapter id.
- `list_loras` reports the active merged adapter id.
- Serving reuse between training steps requires pausing generation across each `start_weight_update` / `update_weights` / `finish_weight_update` session and invalidating caches after sync.
- Models declare merged-LoRA support through `SupportsMergedLoRA` (`merged_lora_spec()` in `vllm_omni/lora/capability.py`). AR stages compose upstream `SupportsLoRA` declarations (`packed_modules_mapping`, `hf_to_vllm_mapper`). Pipelines that do not yet support merged LoRA or are runtime-only are registered explicitly in `vllm_omni/lora/capability.py` with a named reason.

## Not supported

- One request applying different adapters to different LLM stages, or any
  adapter on a downstream LLM stage.
- Enabling runtime vLLM LoRA (`enable_lora=True`) on several LLM stages of a multi-stage pipeline with a
  single global option, or combining runtime `enable_lora=True` with `omni_lora_ipc` on the same stage.
- Sharing an adapter cache between an LLM stage and a diffusion stage.
- Runtime LoRA (`enable_lora=True`) on an LLM stage whose loaded model class does not implement `SupportsLoRA`.
- Merged LoRA on expert-routed MoE layers, quantized linears, HSDP, or inference weight offloading.
- Applying a request's adapter to its CFG companion requests, or to the decode
  stage of a prefill-decode disaggregated pipeline.
- Stacking several adapters in one diffusion request in merged mode.
