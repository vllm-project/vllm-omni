# Diffusion Disaggregate Inference

This guide explains the diffusion-specific split used by vLLM-Omni multi-stage
inference. It follows the generic stage wiring model in
[Disaggregated Inference](disaggregated_inference.md), but it does not add a
new connector transport path for diffusion tensors.

## Overview

A diffusion pipeline can be split into multiple `stage_type=diffusion` stages.
Each stage owns one model component and passes an intermediate payload to the
next stage.

The typical linear shape is:

```text
conditioning -> denoising -> decoding
```

Denoising is also a diffusion component. The name `submodule` in
`worker_type=submodule` is a runtime worker selector, not a universal taxonomy
for all diffusion model components.

## Runtime Model

The current implementation reuses the existing diffusion stage family.

| Selector | Current behavior |
| :--- | :--- |
| `stage_type=diffusion` | The stage uses the diffusion request/output contract. |
| `worker_type=submodule` | The stage runs through `StageSubModuleClient` / `StageSubModuleProc` and calls a direct component entry point. |
| unset `worker_type` | The stage runs through `StageDiffusionClient` / `StageDiffusionProc` and the normal diffusion runner path. |

This keeps `StagePool` and orchestrator routing unchanged. The worker selector
only changes how a diffusion stage process executes its component.

Use the direct submodule worker for lightweight one-shot components. Use the
normal diffusion worker when the component needs runner behavior such as the
denoising loop, scheduler execution, warmup, or batching support.

## Configuration Model

A disaggregated diffusion pipeline is configured as normal stage config. The
model chooses the meaning of `model_stage`.

```yaml
stage_args:
  - stage_id: 0
    stage_type: diffusion
    worker_type: submodule
    engine_args:
      model_stage: conditioning
    final_output: false

  - stage_id: 1
    stage_type: diffusion
    engine_args:
      model_stage: denoising
    engine_input_source: [0]
    custom_process_input_func: pkg.stage_input_processors.model.conditioning_to_denoising
    final_output: false

  - stage_id: 2
    stage_type: diffusion
    worker_type: submodule
    engine_args:
      model_stage: decoding
    engine_input_source: [1]
    custom_process_input_func: pkg.stage_input_processors.model.denoising_to_decoding
    final_output: true
```

The stage names are model-owned. A model may use names such as `encode`,
`denoise`, and `decode`, or a different split, as long as the same stage output
and input conversion contracts are implemented.

## Data Flow

Current diffusion intermediate payloads use the existing request/output path:

```text
stage component
  -> DiffusionOutput.multimodal_output
  -> OmniRequestOutput.multimodal_output
  -> custom_process_input_func
  -> OmniTokensPrompt.additional_information
  -> next diffusion stage
```

The orchestrator does not interpret diffusion tensor schemas. It forwards the
previous stage output to the configured `custom_process_input_func`; the model
stage input processor validates the payload and builds the next prompt.

This feature does not currently move diffusion intermediate tensors through a
connector-backed data plane. Connector configuration remains part of the
generic disaggregated inference layer, but the supported diffusion split here
uses `multimodal_output` and `additional_information` for the handoff.

## Pipeline Contract

A pipeline that supports diffusion disaggregation should provide:

- A component selector, usually `model_stage`, to load and execute the current
  component.
- Component entry points for direct submodule stages.
- A normal diffusion runner path for denoising or other runner-owned work.
- Intermediate outputs in `DiffusionOutput.multimodal_output` for non-final
  stages.
- Stage input processors that validate required keys and build the next
  `OmniTokensPrompt`.
- Stage-aware warmup or dummy inputs for partial-pipeline execution.

Keep model-specific tensor schemas inside the pipeline and stage input
processor. The orchestrator should stay schema-agnostic.

## Reference Implementation

The first in-tree implementation is Qwen-Image. It demonstrates the generic
pattern, but the feature is not limited to Qwen-Image.

Reference config:

`vllm_omni/model_executor/stage_configs/qwen_image_3stage.yaml`

| Stage | Role | Runtime worker | `model_stage` |
| :--- | :--- | :--- | :--- |
| 0 | conditioning / encode | `worker_type=submodule` | `encode` |
| 1 | denoising | normal diffusion worker | `denoise` |
| 2 | decoding / render | `worker_type=submodule` | `decode` |

The single-stage Qwen-Image path remains available through
`model_stage=diffusion`.

## Operational Notes

- Keep split diffusion components under `stage_type=diffusion`.
- Add `worker_type=submodule` only for stages that can run without the normal
  diffusion runner loop.
- Keep payload conversion in `custom_process_input_func`.
- Keep payload validation in the model-specific stage input processor.
- Add focused unit tests for every new stage boundary.
- Add one e2e test for each supported model topology.

## Future Work

- Connector-backed transfer for large diffusion intermediate tensors.
- More reference diffusion pipelines beyond the current Qwen-Image topology.
- Non-linear diffusion stage graphs when a supported model requires them.
