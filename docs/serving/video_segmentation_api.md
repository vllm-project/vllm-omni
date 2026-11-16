# Video Segmentation API

`POST /v1/videos/segmentations` finds every instance of one or more text
concepts in a video and tracks each one through all frames: every instance
keeps one object id, and every frame lists the visible instances with a mask,
a pixel box and a score. It is a vLLM-Omni extension (OpenAI has no
segmentation endpoint) and is available only when the served model declares
video segmentation support; other models answer 400.

Supported models: [SAM 3](../models/supported_models.md) (`facebook/sam3`),
from its Transformers-format `Sam3VideoModel` checkpoint, which carries the
detector and the memory tracker. Checkpoints without Transformers-format
tracker weights, such as SAM 3.1's multiplex release, serve
[image segmentation](image_segmentation_api.md) only. See the
[SAM 3 video recipe](gh-file:recipes/facebook/SAM3-Video-H200.md).

## Quick Start

```bash
vllm serve facebook/sam3 --omni --port 8091
```

`facebook/sam3` is gated: request access on the model page and export
`HF_TOKEN` before starting the server.

Track every person in a clip:

```bash
curl -s http://localhost:8091/v1/videos/segmentations \
  -F "video=@dance.mp4" \
  -F "prompt=person" > people.json
```

Track several concepts at once and also get one mask video per object:

```bash
curl -s http://localhost:8091/v1/videos/segmentations \
  -F "video=@street.mp4" \
  -F "prompt=person" \
  -F 'prompts=["car", "bicycle"]' \
  -F "mask_video=true" > street.json
```

Decode the masks in Python:

```python
import base64

import numpy as np

result = response.json()
height, width = result["height"], result["width"]


def decode(rle):
    runs = np.repeat(np.arange(len(rle["counts"])) % 2 == 1, rle["counts"])
    return runs.reshape((height, width), order="F")  # True inside the instance


for frame in result["frames"]:
    for instance in frame["instances"]:
        mask = decode(instance["mask"])
        print(frame["frame_index"], instance["id"], instance["box"], mask.sum())

for obj in result["objects"]:
    if obj["mask_video"]:
        with open(f"object_{obj['id']}.mp4", "wb") as f:
            f.write(base64.b64decode(obj["mask_video"]))
```

## Request

Multipart form fields:

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `video` | file | — | Input video upload (required, at most 512 MiB). The endpoint does not fetch remote URLs. |
| `prompt` | string | `""` | Concept to track, e.g. `"person"`. |
| `prompts` | JSON string | — | More concepts, e.g. `["car", "bicycle"]`; at most 16 distinct prompts in total. |
| `mask_video` | bool | `false` | Also return each object's mask over the whole video as an H.264 MP4. |
| `model` | string | served model | Must match the served model when given. |

At least one non-empty prompt is required. Every frame of the upload is
processed in decode order. A video may have at most 2048 frames (about 68 s at
30 fps); longer videos answer 400, so split them into clips.

## Response

```json
{
  "created": 1793201902,
  "model": "facebook/sam3",
  "width": 1280,
  "height": 720,
  "num_frames": 270,
  "fps": 30.0,
  "objects": [
    {"id": 0, "prompt": "person", "mask_video": null}
  ],
  "frames": [
    {
      "frame_index": 0,
      "instances": [
        {"id": 0, "score": 0.93, "box": [118.0, 35.0, 412.0, 561.0], "mask": {"size": [720, 1280], "counts": [...]}}
      ]
    }
  ]
}
```

- `objects` lists every object id that is visible on at least one frame, with
  the prompt whose concept it matched.
- `frames` has one entry per input frame. An instance's `score` is its
  detection confidence when it was first found; `box` is `[x0, y0, x1, y1]` in
  input-video pixels.
- Each `mask` is COCO uncompressed RLE: column-major run lengths that start
  with a background run. Masks of objects from the same prompt never overlap.
- With `mask_video=true`, each object's `mask_video` is a base64 H.264 MP4 at
  the input frame rate (24 fps when the container reports none): white inside
  the object, black elsewhere and on frames where it is not visible. Videos
  with an odd width or height gain one black row or column.

## How tracking works

On every frame, the detector finds candidates for each prompt and the tracker
propagates the existing objects from a memory of recent frames. A detection
that overlaps no tracked object starts a new object; detections confirm and
periodically re-anchor the objects they overlap. During an object's first 15
frames it is dropped again if detections stop confirming it or if it
duplicates an earlier object, so results are delayed by 15 frames internally
and such short-lived false starts never appear in the response.

## Offline Inference

The same pipeline runs through `Omni.generate` with the video in
`multi_modal_data["video"]` (a file path, a `[T, H, W, 3]` uint8 array or a
list of PIL images) and extra prompts in
`OmniDiffusionSamplingParams.extra_args["prompts"]`. The result's
`multimodal_output["video_segmentation"]` holds `width`, `height`,
`num_frames`, `fps`, `objects` (object id → prompt) and `frames`, where each
frame has parallel `object_ids`, `scores`, `boxes` and `rle` (COCO counts)
lists.
