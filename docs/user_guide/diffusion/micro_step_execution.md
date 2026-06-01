# Micro-Step Execution

Micro-step execution is an opt-in diffusion execution mode enabled with
`stream_batch=True` when constructing `Omni`. It runs *temporal pipeline
parallelism* on streaming chunked diffusion: at each tick every PP rank
denoises a different chunk at a different timestep, then chunks shift one
rank downstream. One tick = one micro-step.

It is not a generic diffusion toggle for every pipeline. Only pipelines that
implement the micro-step contract support it today.

## Quick Start

```python
import PIL.Image
import numpy as np

from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="lingbot_world/lingbot-world-base-cam/Lingbot-World-Fast",
    model_class_name="LingbotWorldFastPipeline",
    stream_batch=True,
    parallel_config=DiffusionParallelConfig(pipeline_parallel_size=4),
    enforce_eager=True,
)

outputs = omni.generate(
    {
        "prompt": "A sweeping cinematic journey along the Great Wall of China",
        "multi_modal_data": {
            "image": PIL.Image.open("anchor.jpg"),
            "camera": {
                "poses": np.load("poses.npy"),
                "intrinsics": np.load("intrinsics.npy"),
            },
        },
    },
    OmniDiffusionSamplingParams(
        height=480,
        width=832,
        num_chunks=20,
        chunk_frames=12,
        num_inference_steps=5,
        slo_fps=16.0,
        slo_max_batch=4,
        extra_args={"session_id": "demo"},
    ),
)
```

## Sampling Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `chunk_frames` | yes | Pixel frames produced per chunk |
| `num_chunks` | yes | Total number of chunks per request. Output frames = `num_chunks * chunk_frames` after VAE decode |
| `num_inference_steps` | yes | Denoising steps per chunk |
| `slo_fps` | no | Frames-per-second target. Enables SLO-adaptive batching that grows or shrinks per-step admission `B` to meet the budget |
| `slo_max_batch` | no, default 8 | Upper bound on per-step admission `B` |

When `slo_fps` is set, the scheduler observes the wall-clock latency of each
micro-step and adjusts `B_target` for the next admission tick. If latency
exceeds the budget, `B` decreases; if it is comfortably under, `B` grows up
to `slo_max_batch`.

## Supported Pipelines

| Pipeline | Example models | Micro-step execution |
|----------|----------------|----------------------|
| `LingbotWorldFastPipeline` | `lingbot_world/lingbot-world-base-cam/Lingbot-World-Fast` | Yes |
| All other diffusion pipelines | — | No |

## Current Limitations

- `max_num_seqs == 1` — exactly one in-flight request per engine.
- `cache_backend` is not supported together with `stream_batch`.
- Unsupported pipelines fail early during model loading.

## When To Use It

Use micro-step execution when:

- The pipeline is built for streaming chunked output (video chunks, audio
  segments) and you want temporal PP to overlap per-chunk denoising across
  ranks.
- You want SLO-aware admission control to keep up with a real-time
  frame-rate budget under variable load.

For single-request stepwise execution without temporal PP, use
[Step Execution](step_execution.md) instead.

For non-streaming PP (memory scaling on a normal diffusion pipeline), see
[Pipeline Parallelism Guide](parallelism/pipeline_parallel.md).

## For Model Authors

If you want to add micro-step execution support to a new diffusion pipeline,
see the implementation guide:
[Diffusion Micro-Step Execution Design](../../design/feature/diffusion_micro_step_execution.md).

The pipeline must already support PP partitioning. See
[Pipeline Parallel Design](../../design/feature/pipeline_parallel.md).
