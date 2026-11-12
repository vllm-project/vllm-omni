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
| Which models | The model class the stage loads must implement vLLM's `SupportsLoRA`; vLLM checks this when the stage engine starts. Examples: BAGEL stage 0 (`OmniBagelForConditionalGeneration`, run on GPU in [#8196](https://github.com/vllm-project/vllm-omni/pull/8196)) and the HunyuanImage-3 AR stage. The Qwen2.5-Omni and Qwen3-Omni pipelines load `Qwen2_5OmniForConditionalGeneration` / `Qwen3OmniMoeForConditionalGeneration` on stage 0, which do not implement it, and neither do talker or code2wav stages. | Pipelines listed in the [Diffusion LoRA guide](../../user_guide/diffusion/lora.md) |
| Enabled by | `enable_lora` on that stage (stage-scoped key in multi-stage pipelines) | `lora_path` / `lora_backend` on the diffusion config |
| Adapters per request | One, on the entry stage only | One (PEFT backend); the distill backend fuses its weights once at initialization |
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
PEFT backend caches adapters and activates one per request from the request's
`LoRARequest`; the distill backend fuses weights once at initialization. See
the [Diffusion LoRA guide](../../user_guide/diffusion/lora.md) for formats,
supported pipelines and checkpoint name mapping
([#8001](https://github.com/vllm-project/vllm-omni/issues/8001)).

## Not supported

- One request applying different adapters to different LLM stages, or any
  adapter on a downstream LLM stage.
- Enabling vLLM LoRA on several LLM stages of a multi-stage pipeline with a
  single global option.
- Sharing an adapter cache between an LLM stage and a diffusion stage.
- LoRA on an LLM stage whose model class does not implement `SupportsLoRA`.
- Applying a request's adapter to its CFG companion requests, or to the decode
  stage of a prefill-decode disaggregated pipeline.
- Stacking several adapters in one diffusion request (proposed in
  [#9058](https://github.com/vllm-project/vllm-omni/pull/9058)).
