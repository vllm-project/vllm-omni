# Text-To-Video

This example demonstrates how to deploy the Wan2.2 text-to-video model for online video generation using vLLM-Omni.

## Start Server

### Basic Start

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8091
```

### Start with Parameters

Or use the startup script:

```bash
bash run_server.sh
```

The script allows overriding:
- `MODEL` (default: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`)
- `PORT` (default: `8091`)
- `BOUNDARY_RATIO` (default: `0.875`)
- `FLOW_SHIFT` (default: `5.0`)
- `CACHE_BACKEND` (default: `none`)
- `ENABLE_CACHE_DIT_SUMMARY` (default: `0`)

## API Calls

### Method 1: Using curl

```bash
# Basic text-to-video generation
bash run_curl_text_to_video.sh

# Or execute directly (OpenAI-style multipart)
curl -s http://localhost:8091/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=33" \
  -F "negative_prompt=色调艳丽 ，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -F "guidance_scale_2=4.0" \
  -F "boundary_ratio=0.875" \
  -F "seed=42" | jq -r '.data[0].b64_json' | base64 -d > wan22_output.mp4
```

## Request Format

### Simple Text-to-Video Generation

```bash
curl -X POST http://localhost:8091/v1/videos \
  -F "prompt=A cinematic view of a futuristic city at sunset"
```

### Generation with Parameters

```bash
curl -X POST http://localhost:8091/v1/videos \
  -F "prompt=A cinematic view of a futuristic city at sunset" \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=33" \
  -F "negative_prompt=low quality, blurry, static" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -F "guidance_scale_2=4.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=5.0" \
  -F "seed=42"
```

## Generation Parameters

| Parameter             | Type   | Default | Description                                      |
| --------------------- | ------ | ------- | ------------------------------------------------ |
| `prompt`              | str    | -       | Text description of the desired video            |
| `negative_prompt`     | str    | None    | Negative prompt                                  |
| `n`                   | int    | 1       | Number of videos to generate                     |
| `width`               | int    | None    | Video width in pixels                            |
| `height`              | int    | None    | Video height in pixels                           |
| `num_frames`          | int    | None    | Number of frames to generate                     |
| `fps`                 | int    | None    | Frames per second for output video               |
| `num_inference_steps` | int    | None    | Number of denoising steps                        |
| `guidance_scale`      | float  | None    | CFG guidance scale (low-noise stage)             |
| `guidance_scale_2`    | float  | None    | CFG guidance scale (high-noise stage, Wan2.2)     |
| `boundary_ratio`      | float  | None    | Boundary split ratio for low/high DiT (Wan2.2)   |
| `flow_shift`          | float  | None    | Scheduler flow shift (Wan2.2)                    |
| `seed`                | int    | None    | Random seed (reproducible)                       |
| `lora`                | object | None    | LoRA configuration                               |
| `extra_body`          | object | None    | Model-specific extra parameters                  |

## Response Format

```json
{
  "created": 1234567890,
  "data": [
    { "b64_json": "<base64-mp4>" }
  ]
}
```

## Extract Video

```bash
# Extract base64 from response and decode to video
cat response.json | jq -r '.data[0].b64_json' | base64 -d > wan22_output.mp4
```
