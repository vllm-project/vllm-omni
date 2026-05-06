# Adding Step Execution Support for Diffusion Pipelines

This guide documents vLLM-Omni's stepwise diffusion contract for model authors
and contributors implementing `step_execution=True` support for a diffusion
pipeline.

For end-user enablement, supported models, and current limitations, see
[Step Execution](../../user_guide/diffusion/step_execution.md).

This document describes the base step-execution contract only. For the
experimental batching policy layered on top of the step-wise path, see
[Continuous Batching for Step-Wise Diffusion](diffusion_continuous_batching.md).

## Current Support Scope

`step_execution` is **not** a generic diffusion toggle. It only works for
pipelines that implement the segmented stateful contract in
[`vllm_omni/diffusion/models/interface.py`](gh-file:vllm_omni/diffusion/models/interface.py).

This page is intentionally author-facing. Treat runtime enablement
(`step_execution=True` in Python or `--step-execution` in serving) as an
opt-in user knob layered on top of the implementation contract below.

Current in-tree support:

| Pipeline | Example models | Step execution |
|----------|----------------|----------------|
| `QwenImagePipeline` | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Yes |
| All other diffusion pipelines | `QwenImageEditPipeline`, `QwenImageEditPlusPipeline`, `QwenImageLayeredPipeline`, GLM-Image, Wan, Flux, etc. | No |

Current engine/runtime limitations:

- Continuous batching with `max_num_seqs > 1` is experimental and documented in
  [Continuous Batching for Step-Wise Diffusion](diffusion_continuous_batching.md).
  Keep `max_num_seqs=1` if you want the older conservative behavior.
- `cache_backend` is not supported in step mode.
- Request-mode extras such as KV transfer are not wired into step mode yet.
- Unsupported pipelines now fail early during model loading instead of failing on the first request.

## Execution Contract

Step mode is driven by four pipeline methods plus the shared mutable request
state object:

- `prepare_encode(state)`: one-time request preparation.
- `denoise_step(state)`: compute the noise prediction for the current step.
- `step_scheduler(state, noise_pred)`: mutate latents and advance step state.
- `post_decode(state)`: decode the final output after denoising is complete.

The state lives in
[`vllm_omni/diffusion/worker/utils.py`](gh-file:vllm_omni/diffusion/worker/utils.py)
as `DiffusionRequestState`. Store request-scoped tensors there, or use
`state.extra` for model-specific fields that do not justify extending the core
dataclass.

The worker-side step loop lives in
[`vllm_omni/diffusion/worker/diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py):

1. `prepare_encode()` runs once for a new request.
2. `denoise_step()` runs every scheduler tick.
3. `step_scheduler()` mutates `state.latents` and advances `state.step_index`.
4. `post_decode()` runs exactly once after `state.denoise_completed` becomes true.

## Recommended Split

When converting an existing request-level `forward()` pipeline, keep the split
strict and mechanical:

| Request-level phase | Stepwise method | What belongs there |
|---------------------|-----------------|--------------------|
| Input validation, prompt encoding, latent init, timestep prep, per-request scheduler creation | `prepare_encode()` | Anything that should happen once per request |
| Transformer forward / noise prediction | `denoise_step()` | Pure denoise computation for the current timestep |
| `scheduler.step(...)` and `step_index += 1` | `step_scheduler()` | Only latent/state mutation for one step |
| VAE decode / postprocess | `post_decode()` | Final decode only |

Keep the stepwise path reusing the same helpers as the request-level path
whenever possible. Reimplementing the denoise loop from scratch is the easiest
way to introduce behavioral drift.

## Qwen-Image Reference

[`pipeline_qwen_image.py`](gh-file:vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py)
is the reference implementation and is split correctly for the current
contract:

- `prepare_encode()` reuses `_prepare_generation_context()` so prompt encoding,
  latent init, timestep creation, CFG setup, and shape bookkeeping stay aligned
  with `forward()`.
- `prepare_encode()` deep-copies `self.scheduler` **after**
  `prepare_timesteps()` so request-specific scheduler state is isolated.
- `denoise_step()` reuses `_build_denoise_kwargs()` plus
  `predict_noise_maybe_with_cfg()`, so sequential CFG, CFG-parallel, and
  non-CFG behavior stay identical to the request-level path.
- `step_scheduler()` only calls
  `scheduler_step_maybe_with_cfg(..., per_request_scheduler=state.scheduler)`
  and increments `state.step_index`.
- `post_decode()` reuses `_decode_latents()`, so the final image decode matches
  the normal `forward()` path.

That decomposition is the target pattern for future models.

## Rules For New Pipelines

- Do not keep request-scoped scheduler state on `self.scheduler`. Copy it into
  `state.scheduler` during `prepare_encode()`.
- Do not mutate `state.step_index` inside `denoise_step()`. Only
  `step_scheduler()` should advance the step.
- Do not decode partial outputs in `denoise_step()` or `step_scheduler()`.
- If the request-level pipeline has condition latents, masks, or edit-specific
  tensors, store them in `state` or `state.extra`, not in global pipeline
  attributes.
- Preserve CFG behavior by sharing the same helper path used by `forward()`.
- Keep `post_decode()` equivalent to the tail of `forward()`.

## Validation Checklist

Before marking a pipeline as `supports_step_execution = True`, verify:

- Stepwise output matches request-level output for the same seed and sampling params.
- Per-request scheduler state is isolated across concurrent requests.
- Abort during denoise does not leak cached state.
- `step_index` reported by `RunnerOutput` matches the scheduler progress.
- CFG-parallel and non-CFG paths both work if the request-level pipeline supports them.

## Related Files

- Contract: [`vllm_omni/diffusion/models/interface.py`](gh-file:vllm_omni/diffusion/models/interface.py)
- State: [`vllm_omni/diffusion/worker/utils.py`](gh-file:vllm_omni/diffusion/worker/utils.py)
- Runner loop: [`vllm_omni/diffusion/worker/diffusion_model_runner.py`](gh-file:vllm_omni/diffusion/worker/diffusion_model_runner.py)
- Scheduler transport: [`vllm_omni/diffusion/sched/interface.py`](gh-file:vllm_omni/diffusion/sched/interface.py)
- Reference pipeline: [`vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py`](gh-file:vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image.py)
