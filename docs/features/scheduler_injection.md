# Diffusion Scheduler Injection

This guide covers injecting a custom diffusion *sampling* scheduler into a
stock pipeline. It is the supported alternative to forking a pipeline when
only the scheduler class needs to change (for example an SDE sampler that
collects log-probs).

This is **not** the stage request scheduler (`engine_args.scheduler_cls` /
`OmniARScheduler`). The Python API and deploy YAML name is
`diffusion_scheduler` so the two are distinguishable.

## Options

Set these on `AsyncOmni` / `OmniEngineArgs`, or under a deploy YAML stage's
`engine_args`. There are no CLI flags (same as `custom_pipeline_args`).

| Engine arg | `OmniDiffusionConfig` field | Meaning |
| --- | --- | --- |
| `diffusion_scheduler` | `scheduler` | Registry name or dotted class path |
| `diffusion_scheduler_kwargs` | `scheduler_kwargs` | Extra kwargs for `from_pretrained()` |

Resolution order inside `build_pipeline_scheduler`:

1. Explicit `scheduler_cls` argument (pipeline-internal)
2. `od_config.scheduler`
3. The pipeline's existing default builder (bit-identical when unset)

## Usage

```python
from vllm_omni.entrypoints.async_omni import AsyncOmni

engine = AsyncOmni(
    model="Qwen/Qwen-Image",
    diffusion_scheduler="my_pkg.FlowMatchSDEDiscreteScheduler",
    diffusion_scheduler_kwargs={},  # optional, forwarded to from_pretrained
)
```

External packages can advertise classes through the empty
`vllm_omni.schedulers` entry-point group declared in `pyproject.toml`:

```toml
[project.entry-points."vllm_omni.schedulers"]
flow_match_sde = "my_pkg:FlowMatchSDEDiscreteScheduler"
```

Then `diffusion_scheduler="flow_match_sde"` resolves through the registry.

## Injected-class contract

The class must satisfy the same diffusers-style contract stock pipelines
already use:

- `step(noise_pred, t, latents, return_dict=False, generator=None)` — with
  `return_dict=False` return a tuple whose first element is the stepped
  latents
- `set_timesteps(num_inference_steps, device=...)` and expose `.timesteps`.
  Keep named parameters (`sigmas` / `timesteps`); a `*args, **kwargs` override
  hides those names and breaks dummy warmup
- `set_begin_index(...)` for pipeline-parallel timestep slicing
- a `.config` attribute with at least `num_train_timesteps`
- deepcopy-safe (step-wise execution deep-copies the scheduler per request)

Construction matches the stock sites: the pipeline passes its resolved
`local_files_only` (typically `os.path.exists(model)`) and `revision` into
`cls.from_pretrained(od_config.model, subfolder="scheduler", ...)`.
For optional stock-specific timestep controls such as Wan's `shift` or
Boogu's `num_tokens`, the pipeline passes a control only when
`set_timesteps` declares that named parameter. A generic `**kwargs` does not
count as support; unsupported optional controls are omitted with a warning.
Pipeline features that require a schedule input, such as LingBot's refiner
`sigmas`, require that named parameter and raise a clear error if it is absent.

## Which pipelines honor the seam

The factory is wired at every stock `from_pretrained` construction site that
this PR covers, including:

- `flux`, `flux_kontext`, `flux2`
- `qwen_image`, `qwen_image_edit`, `qwen_image_edit_plus`, `qwen_image_layered`
- `sd3`
- `ltx2` (dynamic-shifting overwrite is skipped + warned when injection is set)
- `wan2_2` and `wan2_2_i2v` (per-request `sample_solver` / `flow_shift` rebuild
  is skipped + warned; `shift` is passed only when explicitly supported)
- `glm_image`, `longcat_image*`, `longcat_video_avatar`
- `hunyuan_video_1_5*`
- `ernie_image`, `boogu_image`, `krea2`, `lingbot_video`

Other pipelines fail at load if the selected pipeline did not construct the
configured scheduler through `build_pipeline_scheduler`. This includes
custom pipelines unless they use the same factory. The load-time check also
rejects a pipeline that replaces the constructed instance before loading
finishes, so the config cannot be silently ignored or overwritten.

Custom pipelines remain supported until the rest of the RL injection surface
is ready; see [Custom Diffusion Pipeline](custom_pipeline.md).
