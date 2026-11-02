# Image Segmentation API

`POST /v1/images/segmentations` returns every instance of a prompted concept
in one image: a mask, a pixel box and a confidence score per instance. It is a
vLLM-Omni extension (OpenAI has no segmentation endpoint) and is available only
when the served model declares segmentation support; other models answer 400.

Supported models: [SAM 3 / SAM 3.1](../models/supported_models.md)
(`facebook/sam3.1`, `facebook/sam3`), from Meta's native checkpoint or a
Transformers export. See the [SAM 3.1 recipe](gh-file:recipes/facebook/SAM3.1-H200.md).

## Quick Start

```bash
vllm serve facebook/sam3.1 --omni --port 8091
```

The `facebook/sam3*` repositories are gated: request access on the model page
and export `HF_TOKEN` before starting the server.

Segment by a text concept:

```bash
curl -s http://localhost:8091/v1/images/segmentations \
  -F "image=@street.jpg" \
  -F "prompt=yellow taxi" > taxis.json
```

Add exemplar boxes, or use boxes alone. Each box is `[x0, y0, x1, y1]` in
input-image pixels; a label of `1` marks a positive example and `0` a negative
one (an object that must not match):

```bash
curl -s http://localhost:8091/v1/images/segmentations \
  -F "image=@street.jpg" \
  -F "prompt=taxi" \
  -F "boxes=[[120, 340, 410, 560], [900, 300, 1100, 480]]" \
  -F "box_labels=[1, 0]" \
  -F "mask_format=rle"
```

Decode a PNG mask in Python:

```python
import base64
import io

from PIL import Image

result = response.json()
for instance in result["data"]:
    mask = Image.open(io.BytesIO(base64.b64decode(instance["mask"])))  # mode "L", 255 inside
    print(instance["score"], instance["box"], mask.size)
```

## Request

Multipart form fields:

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `image` | file | — | Input image upload (required). The endpoint does not fetch remote URLs. |
| `prompt` | string | `""` | Concept to segment, e.g. `"red car"`. May be empty when `boxes` are given. |
| `boxes` | JSON string | — | Exemplar boxes `[[x0, y0, x1, y1], ...]` in pixels, at most 64. |
| `box_labels` | JSON string | all `1` | One label per box: `1` positive, `0` negative. |
| `score_threshold` | float | `0.5` | Keep instances whose score exceeds this value, in `[0, 1]`. |
| `mask_threshold` | float | `0.5` | Binarization threshold of the mask probabilities, in `[0, 1]`. |
| `mask_format` | string | `png` | `png` (base64 single-channel PNG) or `rle` (COCO uncompressed RLE). |
| `model` | string | served model | Must match the served model when given. |

At least one of `prompt` and `boxes` is required.

## Response

```json
{
  "created": 1793201902,
  "model": "facebook/sam3",
  "width": 1280,
  "height": 720,
  "mask_format": "png",
  "data": [
    {"score": 0.93, "box": [118.2, 335.0, 412.7, 561.9], "mask": "iVBORw0KGgo..."}
  ]
}
```

Instances are ordered by descending `score`. Masks and boxes are in the input
image's resolution. With `mask_format=rle` each `mask` is
`{"size": [height, width], "counts": [...]}`: column-major run lengths that
start with a background run, as in COCO's uncompressed RLE.

## Offline Inference

The same pipeline runs through `Omni.generate` with the image in
`multi_modal_data` and the box prompts and thresholds in
`OmniDiffusionSamplingParams.extra_args` (`boxes`, `box_labels`,
`score_threshold`, `mask_threshold`). The result's
`multimodal_output["segmentation"]` holds bit-packed `masks`
(`numpy.packbits` over each flattened mask), `mask_shape`
(`[count, height, width]`), `boxes` and `scores`.
