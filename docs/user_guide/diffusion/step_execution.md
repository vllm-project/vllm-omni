# Step Execution

Step execution is an opt-in diffusion execution mode enabled with
`step_execution=True` when constructing `Omni`.

It is not a generic diffusion toggle for every pipeline. Only pipelines that
implement the stepwise contract support it today.

## Quick Start

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="Qwen/Qwen-Image",
    step_execution=True,
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
    ),
)
```

## Supported Pipelines

| Pipeline | Example models | Step execution |
|----------|----------------|----------------|
| `QwenImagePipeline` | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Yes |
| All other diffusion pipelines | `QwenImageEditPipeline`, `QwenImageEditPlusPipeline`, `QwenImageLayeredPipeline`, GLM-Image, Wan, Flux, etc. | No |

## Current Limitations

- `step_execution` currently supports `batch_size=1` only.
- `cache_backend` is not supported together with step execution.
- Unsupported pipelines fail early during model loading.
- Request-mode extras such as KV transfer are not wired into step mode yet.

## When To Use It

Use step execution only when you specifically need the pipeline to run through
its stepwise request state machine. For normal diffusion inference, leave it
disabled unless your workflow depends on this mode.

If you are looking for general diffusion speedups, see
[Diffusion Acceleration Overview](../diffusion_acceleration.md).

## Troubleshooting

If model loading fails with a message mentioning `prepare_encode()`,
`denoise_step()`, `step_scheduler()`, and `post_decode()`, the selected
pipeline does not support step execution.

## For Model Authors

If you want to add step execution support to a new diffusion pipeline, see the
implementation guide:
[Diffusion Step Execution Design](../../design/feature/diffusion_step_execution.md).
