# World Model Repository Map

Resolve these paths and their callers in the target checkout before copying
an interface. The case studies describe historical revisions, while current
source and tests define the integration API.

| Concern | Starting point |
| --- | --- |
| Contribution and model rules | `docs/contributing/README.md`, `docs/contributing/model/README.md`. |
| Discovery and aliases | `vllm_omni/diffusion/utils/hf_utils.py`, `model_metadata.py`, `registry.py`. |
| Components and weights | `vllm_omni/diffusion/model_loader/`, adjacent `load_weights` and `WeightsMapper` implementations. |
| Component, step, and interaction protocols | `vllm_omni/diffusion/models/interface.py`. |
| AR capabilities and KV geometry | `vllm_omni/experimental/ar_diffusion/capability.py`. |
| Session transactions and execution | The same directory's `tick_protocol.py`, `session.py`, `consumer.py`, `runner.py`, and `engine.py`. |
| Physical KV storage | The same directory's `kv_cache/{manager,state,segmented,paged_attention}.py`. |
| Step scheduling | `vllm_omni/diffusion/sched/step_scheduler.py` and worker/model runner. |
| Public model parameters | `vllm_omni/model_extras/` and `vllm_omni/diffusion/interaction/`. |
| Deployments | `vllm_omni/deploy/`, `vllm_omni/config/pipeline_registry.py`. |
| Video and A/V delivery | `vllm_omni/entrypoints/openai/serving_video*.py`, `video_api_utils.py`, and the selected codec implementation. |
| Shared examples | `examples/offline_inference/{text_to_video,diffusion}/`, `examples/online_serving/streaming_video_generation/`. |

Read the branch-local
[AR capability design](../../../../docs/design/ar_diffusion_pipeline_capability.md)
and [realtime design](../../../../docs/design/feature/realtime_ar_diffusion.md)
alongside the source when changing state ownership or serving behavior.

## Search commands

```bash
rg -n 'SupportsARDiffusionPipeline|SupportsStepExecution|SupportsInteractionApply' vllm_omni
rg -n 'ar_diffusion_kv_cache_spec|bind_ar_diffusion_state|close_ar_diffusion_session' vllm_omni
rg -n 'prepare_encode|prepare_next_chunk|denoise_step|step_scheduler|post_decode' vllm_omni/diffusion/models
rg -n 'model_owned_state_bytes_per_session|segmented|session_capacity' vllm_omni/experimental/ar_diffusion
rg --files tests examples recipes | rg 'lingbot_world|abot_world|taomate_h3|hy_nomas'
```

## Reuse contracts

- Resolve the current precedence of standard indexes, modular indexes, and
  native detection. A native checkpoint need not gain a synthetic index.
- Use AR capability hooks for KV geometry, temporary binding, reset, and close;
  use step and interaction protocols for execution boundaries.
- Declare component discovery for DiT, encoders, and VAEs, and verify that
  offload resolves the actual module paths.
- Reuse established worker groups and ranks. An explicitly permitted reference
  adapter can bridge its SP implementation to those groups without owning
  group creation or destruction.
- Normalize native LoRA serialization into the existing manager when possible;
  keep model-specific layout and teacher/student policies in the model package.

## Test routing

| Behavior | Test location |
| --- | --- |
| Model math, loading, controls, geometry | `tests/diffusion/models/{model}/`. |
| KV, sessions, worker lifecycle | `tests/diffusion/ar_diffusion/`. |
| Registry, public parameters, codecs | `tests/model_extras/`, `tests/entrypoints/`, and existing discovery tests. |
| Independent reference | `tests/e2e/accuracy/`. |
| Offline and serving E2E | `tests/e2e/{offline_inference,online_serving}/`. |

Use [vllm-omni-test](../../vllm-omni-test/SKILL.md) to select current markers,
helpers, and Buildkite routing. Recipe setup, usage, and qualified limitations
follow [add-recipe](../../add-recipe/SKILL.md).
