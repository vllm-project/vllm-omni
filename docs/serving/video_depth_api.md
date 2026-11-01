# Video Depth API

`POST /v1/videos/depth` estimates a temporally consistent depth map for every
frame of a video. It returns the depth as a rendered H.264 video and, on
request, as raw per-pixel values. It is a vLLM-Omni extension (OpenAI has no
depth endpoint) and is available only when the served model declares video
depth support; other models answer 400.

Supported models: [Video Depth Anything](../models/supported_models.md)
(`depth-anything/Video-Depth-Anything-{Small,Base,Large}` for relative depth,
`depth-anything/Metric-Video-Depth-Anything-{Small,Base,Large}` for metric
depth). See the
[Video Depth Anything recipe](gh-file:recipes/depth-anything/Video-Depth-Anything-H200.md).

## Quick Start

```bash
vllm serve depth-anything/Video-Depth-Anything-Small --omni --port 8091
```

Get a grayscale depth video (relative models: near is bright; metric models: far is bright):

```bash
curl -s http://localhost:8091/v1/videos/depth -F "video=@walk.mp4" \
  | jq -r .video | base64 -d > walk_depth.mp4
```

Get a colored video and the raw depth:

```python
import base64
import io

import numpy as np
import requests

with open("walk.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8091/v1/videos/depth",
        files={"video": f},
        data={"colormap": "inferno", "return_depth": "true"},
    )
result = response.json()
with open("walk_depth.mp4", "wb") as f:
    f.write(base64.b64decode(result["video"]))
depth = np.load(io.BytesIO(base64.b64decode(result["depth"])))["depth"]  # [T, H, W] float16
```

## Request

Multipart form fields:

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `video` | file | — | Input video upload (required, at most 512 MiB). The endpoint does not fetch remote URLs. |
| `input_size` | int | `518` | Shorter side of the network input, a multiple of 14 from 14 to 1022. Larger values keep finer detail and cost more memory and time. |
| `max_res` | int | `1280` | Frames whose longer side exceeds this are downscaled first; the output has the downscaled size. `-1` keeps the input resolution. |
| `colormap` | string | `grayscale` | Rendering of the depth video: `grayscale` or `inferno`. |
| `return_depth` | bool | `false` | Also return the raw depth as a base64 NPZ archive. |
| `model` | string | served model | Must match the served model when given. |

Every frame of the upload is processed in decode order. A video may have at
most 2048 frames (about 68 s at 30 fps); longer videos answer 400, so split
them into clips.

## Response

```json
{
  "created": 1793546102,
  "model": "depth-anything/Video-Depth-Anything-Small",
  "width": 1280,
  "height": 720,
  "num_frames": 195,
  "fps": 30.0,
  "metric": false,
  "depth_min": 0.0,
  "depth_max": 9.53,
  "video": "<base64 MP4>",
  "depth": null
}
```

- `width`, `height` are the output size: the input size, or its downscaled size
  when the longer side exceeds `max_res`.
- `metric` tells how to read the values. Relative models predict affine-invariant
  inverse depth (disparity): larger is nearer, and the scale is consistent
  across the video but has no unit. Metric models predict depth in meters:
  larger is farther.
- `video` is an H.264 MP4 at the input frame rate (24 fps when the container
  reports none). Values are normalized once over the whole video, from
  `depth_min` to `depth_max`, so brightness is comparable across frames.
  Videos with an odd width or height gain one black row or column.
- With `return_depth=true`, `depth` is a base64 NPZ archive whose `depth` entry
  is a `[num_frames, height, width]` float16 array.

## How long videos are processed

The model sees 32 frames at a time. Consecutive windows advance by 22 frames,
and each later window also re-reads keyframes of the previous one. For relative
models, each window is aligned to the previous one with a least-squares scale
and shift over the shared keyframes; for every model, the 8 overlapping frames
are cross-faded. Memory therefore does not grow with the video length apart
from the decoded frames and the output.

## Offline Inference

The same pipeline runs through `Omni.generate` with the video in
`multi_modal_data["video"]` (a file path, a `[T, H, W, 3]` uint8 RGB array or a
list of PIL images) and the options `input_size` and `max_res` in
`OmniDiffusionSamplingParams.extra_args`. The result's
`multimodal_output["video_depth"]` holds `depth` (a `[T, H, W]` float16 array),
`fps`, `width`, `height`, `num_frames` and `metric`.
