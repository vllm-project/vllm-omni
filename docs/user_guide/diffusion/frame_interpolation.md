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

## Request Parameters

The video APIs `/v1/videos` and `/v1/videos/sync` accept:

| Parameter | Type | Default | Description |
| ----------- | ------ | --------- | ------------- |
| `enable_frame_interpolation` | bool | `false` | Enable post-generation frame interpolation |
| `frame_interpolation_exp` | int | `1` | Interpolation exponent. `1=2x`, `2=4x`, etc. |
| `frame_interpolation_scale` | float | `1.0` | RIFE inference scale |
| `frame_interpolation_model_path` | str | `None` | Local directory or Hugging Face repo ID containing `flownet.pkl` |

## Execution Flow

For supported Wan2.2 pipelines, the execution order is:

1. Diffusion worker finishes denoising and decodes the raw video tensor.
2. Worker-side model-specific post-processing runs.
3. If frame interpolation is enabled, RIFE interpolates the decoded video
   tensor on the worker side and records a FPS multiplier in
   `metadata.video.video_fps_multiplier`.
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

## Python resampling with an explicit output clock

Custom worker-side postprocessors can use `resample_video_tensor` to stretch
a CFR visual sequence onto a specified output timeline. For example, 192
frames at 24 FPS can occupy 15 seconds at 48 FPS, producing exactly 720 frames.
The existing exponent API would instead produce 383 frames at 48 FPS.

`FrameInterpolationPlan` accepts integer or `fractions.Fraction` clocks, derives
the output count from `output_fps * output_duration`, and rejects nonintegral
counts. The source duration is `source_frames / source_fps`. Each output slot
`j` has presentation time `j / output_fps` and samples source coordinate
`j * source_frames / output_frames`. Fractional coordinates use RIFE's pair
timestep; exact source positions are copied. The last source frame is held
through the final source frame's exposure interval. This samples a new grid;
it does not preserve every original frame timestamp or reconstruct missing
actions.

The input must be `[batch, 3, frames, height, width]`, with positive dimensions.
Supported values are `uint8` in `[0, 255]` or `float16`, `bfloat16`, `float32`
in `[0, 1]`. Layout and signed floating ranges are not inferred by this API.
The result has the same layout and dtype on the selected model execution
device. A CPU input may be transport state; it does not force CPU execution.

The following small synthetic example encodes using the plan's **absolute**
FPS. Its float32 output needs 8,847,360 bytes (about 8.4 MiB); the explicit
16 MiB output budget below is for this 32×32 example.

```python
from fractions import Fraction

import av
import torch

from vllm_omni.diffusion.postprocess.frame_timing import FrameInterpolationPlan
from vllm_omni.diffusion.postprocess.rife_interpolator import resample_video_tensor

video = torch.rand(1, 3, 192, 32, 32)  # Synthetic source, not model output.
plan = FrameInterpolationPlan(192, Fraction(24), Fraction(48), Fraction(15))
output = resample_video_tensor(
    video,
    plan,
    model_path="/path/to/RIFE-4.22.lite",
    max_output_bytes=16 * 1024**2,
)
rgb = (output[0].permute(1, 2, 3, 0).float().clamp(0, 1) * 255).round()
rgb = rgb.to(dtype=torch.uint8, device="cpu").numpy()
with av.open("resampled.mp4", "w") as container:
    stream = container.add_stream("libx264", rate=plan.output_fps)
    stream.width, stream.height = rgb.shape[2], rgb.shape[1]
    stream.pix_fmt = "yuv420p"
    for index, array in enumerate(rgb):
        frame = av.VideoFrame.from_ndarray(array, format="rgb24")
        frame.pts = index
        frame.time_base = 1 / plan.output_fps
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
```

The initial interface requires at least two source frames and supports
expansion up to 4096 output frames. It rejects downsampling and does not
interpret variable-frame-rate timestamps. `max_output_bytes` defaults to
1 GiB and limits the float32 output allocation before normalization, model
loading or device transfer. This is **not** a bound on total memory: model
weights, input/normalization tensors, inference workspaces, output dtype
conversion and encoding require additional capacity.

Budget the actual output shape before increasing the resolution:

```text
float32 output bytes = batch * 3 * output_frames * height * width * 4
```

For the same 720-frame plan with batch size one:

| Resolution | Float32 output allocation |
| ------------ | --------------------------- |
| 32×32 | 8.4 MiB |
| 832×480 | 3.21 GiB |
| 1088×1920 | 16.81 GiB |

Both larger examples exceed the default 1 GiB cap. After provisioning enough
memory for the output **and** the other allocations above, pass an explicit
budget, such as `max_output_bytes=4 * 1024**3` for the 832×480 output. Raising
this limit does not reserve memory or guarantee that inference will fit. Batch
size multiplies the output cost, and lower-precision input still uses a
float32 output buffer during interpolation.

An optional `check_cancel` callback can raise at slot boundaries. The helper
propagates cancellation and inference errors without changing the clock or
retrying diffusion. The caller owns device-work completion and worker-lane
cleanup; the helper adds no distributed cancellation collective.

This is a Python postprocessing interface, not a new `/v1/videos` parameter.
Callers must encode at the plan's absolute FPS; do not pass the tensor through
the existing integer FPS-multiplier path. Audio, audio/video synchronization,
scene-cut detection and complete audio-first workflows are outside this helper.
The existing exponent API and server behavior are unchanged.
