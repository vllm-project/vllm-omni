# PAN 2

> Text-to-video and image-to-video generation with PAN 2

## Summary

- Vendor: IFM
- Model: [`IFM/PAN2`](https://huggingface.co/IFM/PAN2)
- Tasks: text-to-video (T2V) and image-to-video from a first frame (I2V)
- Mode: offline `Omni.generate` and OpenAI-compatible `/v1/videos` serving
- Hardware: see [Hardware profiles](#hardware-profiles)
- Maintainer: IFM

## Overview

**PAN 2** is a foundation world model that simulates how a scene evolves under a
time-grounded plan of actions and camera movements. Its 29B generative
simulator is a dual-stream multimodal diffusion transformer (MMDiT): the prompt
is encoded into a continuous state stream, and the state and video streams,
each with its own projections and feed-forward networks, are joined by
self-attention at every layer. The plan guides video denoising, while the
developing video refines the state. One server handles both tasks: a request
with a first-frame image runs image-to-video (I2V), a request without one runs
text-to-video (T2V).

| Item | Contract |
| --- | --- |
| Tasks | T2V; I2V from one first-frame image |
| Prompt | One structured-caption prompt per request, plus an optional negative prompt |
| Image input | Exactly one image for I2V, scaled to cover the requested size and center cropped |
| Default output | 704x1248, 313 frames (13 s) at 24 fps |
| Longest output | 481 frames (20 s) at 24 fps |
| Size | `height` and `width` multiples of 32 |
| Frame count | `num_frames` of the form `4k+1` (`duration_sec * 24 + 1` for whole seconds) |
| Sampling | 50 steps and CFG 2.5 by default; PAN 2's own flow-matching schedule |
| Video input | Not supported |

## Prompt format

PAN 2 needs a structured prompt. It is trained on the captions written by its
[reasoning backbone](https://huggingface.co/IFM/PAN2#reasoning-backbone), and
expects the prompt as that caption serialized to
**compact one-line JSON** (`json.dumps(caption, ensure_ascii=False,
separators=(",", ":"))`). A plain one-line prompt produces unusable video. The
caption describes the style, lighting, camera, subjects, the first frame, and a
timeline of events:

```json
{
  "video_style": "Serene aerial travel footage during blue hour, captured in one uninterrupted continuous shot.",
  "lighting": "Soft blue skylight after sunset, a pale violet horizon and small pools of amber light ...",
  "perspective_view": "Wide drone view looking obliquely down along Santorini's caldera rim ...",
  "subjects_lifecycle": [],
  "init_frame_visual_description": "Two cobalt-blue church domes rise among tightly stepped white Cycladic buildings ...",
  "events": [
    {
      "event_index": 0,
      "time_range_sec": "0 to 4",
      "event_description": "The drone glides gently right along the cliff ...",
      "camera_motion": [{"time_range_sec": "0 to 4", "camera_motion": ["track_right"]}],
      "contained_actions": []
    }
  ],
  "quality": {"video_quality": "Resolved architectural edges ...", "text_logos_overlays": null, "artifacts": null}
}
```

The negative prompt uses the same format. Without one, or with an empty one,
PAN 2 uses the negative prompt it was released with, its time ranges set to
the clip's duration. The examples below read the prompt and negative prompt
from `prompt.json` and `negative_prompt.json` (`i2v_prompt.json` and
`i2v_negative_prompt.json` for image-to-video).

## Guardrails

PAN 2 runs guardrails on the prompt and the generated video, and they may block
a request with HTTP 400. They need the `pan2-guardrail` package:

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail
```

## Hardware profiles

Every profile is sized for the longest 481-frame, 704x1248 output. Peaks are
per GPU.

| GPU memory | GPUs | Key arguments | Peak memory | Recipe |
| --- | ---: | --- | ---: | --- |
| ≥141 GB (H200, B200, B300) | 1 | (none) | 107.4 GiB | This page |
| | 4 | `--cfg-parallel-size 2 --usp 2` | 110.7 GiB | This page |
| | 8 | `--cfg-parallel-size 2 --usp 4` | 110.4 GiB | This page |
| 96 GB (RTX PRO 6000) | 1 | `--enable-distributed-layerwise-offload --dlo-resident-layers 44`, `offload_guardrail_models` | 87.0 GiB | [PAN2-RTX-PRO-6000](PAN2-RTX-PRO-6000.md) |
| | 2 | `--tensor-parallel-size 2` | 87.0 GiB | [PAN2-RTX-PRO-6000](PAN2-RTX-PRO-6000.md) |
| 32 GB (RTX 5090) | 1 | `--enable-distributed-layerwise-offload --dlo-resident-layers 0`, `offload_guardrail_models` | 29.8 GiB | [PAN2-RTX-5090](PAN2-RTX-5090.md) |
| | 2 | `--tensor-parallel-size 2 --enable-distributed-layerwise-offload --dlo-resident-layers 6`, `offload_guardrail_models` | 29.5 GiB | [PAN2-RTX-5090](PAN2-RTX-5090.md) |

`offload_guardrail_models` is set with
`--stage-overrides '{"0": {"model_config": {"offload_guardrail_models": true}}}'`.
With `--enable-distributed-layerwise-offload`, the DiT blocks stay in pinned
host memory and stream to the GPU each step, with `--dlo-resident-layers`
leading blocks kept on the GPU. The offloaded weights live in pinned system
RAM, so these profiles need a large amount of it, and each CFG-parallel branch
adds its own copy. Recipes for 80 GB ([PAN2-H100](PAN2-H100.md)) and 48 GB
([PAN2-L40S](PAN2-L40S.md)) GPUs are also available.

On ≥141 GB GPUs:

```bash
pip install git+https://github.com/ifm-ai/pan2-guardrail  # PAN 2 guardrails

# 1 GPU
vllm serve IFM/PAN2 --omni --port 8000

# 4 GPUs
vllm serve IFM/PAN2 --omni --port 8000 \
  --num-gpus 4 --cfg-parallel-size 2 --usp 2 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile

# 8 GPUs
vllm serve IFM/PAN2 --omni --port 8000 \
  --num-gpus 8 --cfg-parallel-size 2 --usp 4 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile
```

## Requests

Text-to-video (704x1248, 241 frames = 10 s at 24 fps):

```bash
curl -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "prompt=<prompt.json" \
  -F "negative_prompt=<negative_prompt.json" \
  -F "size=1248x704" \
  -F "num_frames=241" \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=2.5" \
  -F "seed=42" \
  --output pan2_t2v.mp4
```

First frame + text to video:

```bash
curl -X POST http://localhost:8000/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "input_reference=@first_frame.jpg" \
  -F "prompt=<i2v_prompt.json" \
  -F "negative_prompt=<i2v_negative_prompt.json" \
  -F "size=1248x704" \
  -F "num_frames=241" \
  -F "fps=24" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=2.5" \
  -F "seed=42" \
  --output pan2_i2v.mp4
```

Use `num_frames` to set the length; `seconds` is not used. Requests with a
size that is not a multiple of 32, a frame count that is not `4k+1`, more than
one image, video input, or custom `sigmas`/`timesteps` are rejected with HTTP
400 before they reach the GPUs. The asynchronous `/v1/videos` endpoint takes
the same fields.

### Offline

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model IFM/PAN2 \
  --prompt "$(cat prompt.json)" --negative-prompt "$(cat negative_prompt.json)" \
  --output pan2_t2v.mp4

python examples/offline_inference/image_to_video/image_to_video.py \
  --model IFM/PAN2 --image first_frame.jpg \
  --prompt "$(cat i2v_prompt.json)" --negative-prompt "$(cat i2v_negative_prompt.json)" \
  --output pan2_i2v.mp4
```

Both scripts accept the parallel and offload flags used in the profiles above,
for example `--cfg-parallel-size 2 --ulysses-degree 2`.

## Request-scoped quality

PAN 2 can switch Cache-DiT per request without a startup option:
`-F 'quality=lossless'` runs the reference path, `-F 'quality=high'` PAN 2's
conservative Cache-DiT profile. `high` is about 1.3x faster than `lossless`;
the speedup depends on the prompt and the parallel configuration.
