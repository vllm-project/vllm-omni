# Step Execution

Step execution is an opt-in diffusion execution mode enabled with
`step_execution=True` when constructing `Omni`.

It is not a generic diffusion toggle for every pipeline. Only pipelines that
implement the stepwise contract support it today.

## Quick Start

### Python API

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

### Serving

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 8
```

For serving, `--step-execution` enables the step-wise runtime. Continuous
batching only becomes relevant when `--max-num-seqs > 1`.

## Supported Pipelines

| Pipeline | Example models | Step execution |
|----------|----------------|----------------|
| `QwenImagePipeline` | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Yes |
| All other diffusion pipelines | `QwenImageEditPipeline`, `QwenImageEditPlusPipeline`, `QwenImageLayeredPipeline`, GLM-Image, Wan, Flux, etc. | No |

!!! warning "Experimental continuous batching"
    When `--step-execution` is enabled and `max_num_seqs > 1` is configured,
    the step-wise path can batch
    compatible requests together. This is experimental. Requests with
    incompatible sampling parameters are intentionally kept in separate batches,
    and `max_num_seqs=1` remains the conservative default.

## Current Limitations

- Continuous batching under `step_execution` is experimental and only batches
  compatible requests.
- `cache_backend` is not supported together with step execution.
- Unsupported pipelines fail early during model loading.
- Request-mode extras such as KV transfer are not wired into step mode yet.

## When To Use It

Use step execution only when you specifically need the pipeline to run through
its stepwise request state machine. For normal diffusion inference, leave it
disabled unless your workflow depends on this mode.

For Qwen-Image online serving, the usual progression is:

- start with `--step-execution --max-num-seqs 1` if you only need the step-wise path
- increase `--max-num-seqs` after that if you want the experimental compatible-request batching behavior

If you are looking for general diffusion speedups, see
[Diffusion Features Overview](../diffusion_features.md).

## Troubleshooting

If model loading fails with a message mentioning `prepare_encode()`,
`denoise_step()`, `step_scheduler()`, and `post_decode()`, the selected
pipeline does not support step execution.

## For Model Authors

If you want to add step execution support to a new diffusion pipeline, see the
implementation guide:
[Diffusion Step Execution Design](../../design/feature/diffusion_step_execution.md).

If you also want that pipeline to participate in the experimental batched
step-wise path, see:
[Continuous Batching for Step-Wise Diffusion](../../design/feature/diffusion_continuous_batching.md).
