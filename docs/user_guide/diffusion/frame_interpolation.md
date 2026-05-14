# Frame Interpolation

## Overview

vLLM-Omni supports post-generation frame interpolation for supported video
diffusion pipelines. This feature inserts synthesized intermediate frames
between adjacent generated frames to improve temporal smoothness without
rerunning the diffusion denoising loop.

Frame interpolation runs in the diffusion worker post-processing path instead
of the API server encoding path. This allows the interpolation step to reuse
the worker's current accelerator device and keeps the FastAPI event loop free
from heavy synchronous PyTorch work.

For an input video with `N` generated frames and interpolation exponent `exp`,
the output frame count is:

```text
(N - 1) * 2**exp + 1
```

The output FPS is multiplied by `2**exp` so the clip duration remains close to
the original generated video.

## Supported Pipelines

Frame interpolation is currently supported for:

- `WanPipeline` (Wan2.2 text-to-video)
- `WanImageToVideoPipeline`
- `Wan22TI2VPipeline`

## Request Parameters

The video APIs `/v1/videos` and `/v1/videos/sync` accept:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_frame_interpolation` | bool | `false` | Enable post-generation frame interpolation |
| `frame_interpolation_exp` | int | `1` | Interpolation exponent. `1=2x`, `2=4x`, etc. |
| `frame_interpolation_scale` | float | `1.0` | RIFE inference scale |
| `frame_interpolation_model_path` | str | `None` | Local directory or Hugging Face repo ID containing `flownet.pkl` |

## Execution Flow

For supported Wan2.2 pipelines, the execution order is:

1. Diffusion worker finishes denoising and decodes the raw video tensor.
2. Worker-side model-specific post-processing runs.
3. If frame interpolation is enabled, RIFE interpolates the decoded video
   tensor on the worker side and records a FPS multiplier in `custom_output`.
4. The API server receives the already-interpolated video and only performs
   MP4 export.

This design keeps interpolation close to the generated tensor and avoids
introducing another heavyweight GPU context in the API server process.

## Example

Start the server:

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8091
```

Run a sync request with interpolation enabled:

```bash
curl -X POST http://localhost:8091/v1/videos/sync \
  -F "prompt=A dog running through a park" \
  -F "num_frames=81" \
  -F "width=832" \
  -F "height=480" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=1.0" \
  -F "guidance_scale_2=1.0" \
  -F "enable_frame_interpolation=true" \
  -F "frame_interpolation_exp=1" \
  -F "frame_interpolation_scale=1.0" \
  -F "seed=42" \
  -o sync_t2v_interpolated.mp4
```

## Notes

- This is a post-processing feature. It does not modify the diffusion denoising
  schedule.
- Higher interpolation exponents increase post-processing time and memory usage.
- If the interpolation model weights are not available locally,
  `frame_interpolation_model_path` may point to a Hugging Face repo containing
  `flownet.pkl`.
