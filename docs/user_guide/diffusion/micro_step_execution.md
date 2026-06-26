# Micro-Step Execution

Micro-step execution is an opt-in diffusion mode enabled with
`stream_batch=True` when constructing `Omni`. It runs *temporal pipeline
parallelism* over streaming chunked diffusion: each micro-step advances a
fixed ladder of `num_inference_steps` slots — every PP rank denoises a chunk at
a different timestep, and the deepest (finished) chunk rolls off for decode.

It is not a generic toggle — only pipelines implementing the micro-step
contract support it.

## Quick Start

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="<causvid-checkpoint-dir>",
    model_class_name="CausVidPipeline",
    stream_batch=True,
    parallel_config=DiffusionParallelConfig(pipeline_parallel_size=4),
    enforce_eager=True,
    enable_dynamic_block_scheduling=True,
)

outputs = omni.generate(
    {"prompt": "A cybernetic bird on a neon rooftop"},
    OmniDiffusionSamplingParams(
        height=480,
        width=832,
        num_chunks=20,
        chunk_frames=4,
        num_inference_steps=4,
        extra_args={
            "session_id": "demo",
            "video_path": "source.mp4",
            "noise_scale": 0.8,
            "local_attn_size": 3,
            "sink_size": 3,
            "sink_threshold": 0.2,
        },
    ),
)
```

## Sampling Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `chunk_frames` | yes | Pixel frames produced per chunk |
| `num_chunks` | yes | Chunks per request; output frames ≈ `num_chunks * chunk_frames` after VAE decode |
| `num_inference_steps` | yes | Denoising steps per chunk = the number of ladder slots |

`extra_args` are pipeline-specific (e.g. CausVid's `video_path` / `noise_scale`).

## Supported Pipelines

| Pipeline | Micro-step execution |
|----------|----------------------|
| `CausVidPipeline` (reference) | Yes |
| All other diffusion pipelines | No |

## Current Limitations

- `max_num_seqs == 1` — exactly one in-flight request per engine.
- `cache_backend` is not supported together with `stream_batch`.
- Unsupported pipelines fail early during model loading.

## When To Use It

Use it for streaming chunked output (video chunks) where temporal PP can
overlap per-chunk denoising across ranks. For single-request stepwise execution
without temporal PP use [Step Execution](step_execution.md); for memory-scaling
PP on a normal pipeline see [Pipeline Parallelism](parallelism/pipeline_parallel.md).

## For Model Authors

See [Diffusion Micro-Step Execution Design](../../design/feature/diffusion_micro_step_execution.md).
The pipeline must already support PP partitioning
([Pipeline Parallel Design](../../design/feature/pipeline_parallel.md)).
