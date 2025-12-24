# Text-To-Video

This example demonstrates how to deploy Wan2.2 text-to-video model for online generation using vLLM-Omni.

## Start Server

### Basic Start

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8093 --boundary-ratio 0.875 --flow-shift 5.0
```

Notes:
- `flow-shift`: 5.0 for 720p, 12.0 for 480p (Wan2.2 recommendation).
- `boundary-ratio`: 0.875 for Wan2.2 low/high DiT split.

### Start with Parameters

Or use the startup script:

```bash
bash run_server.sh
```

## API Calls

### Method 1: Using curl

```bash
# Basic text-to-video generation
bash run_curl_text_to_video.sh

# Or execute directly
curl -s http://localhost:8093/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A cinematic shot of a flying kite over the ocean."}
    ],
    "extra_body": {
      "height": 720,
      "width": 1280,
      "num_frames": 81,
      "fps": 24,
      "num_inference_steps": 40,
      "guidance_scale": 4.0,
      "guidance_scale_2": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].video_url.url' | cut -d',' -f2 | base64 -d > output.mp4
```

### Method 2: Using Python Client

```bash
python openai_chat_client.py --prompt "A cinematic shot of a flying kite over the ocean." --output output.mp4
```

## Request Format

### Simple Text Generation

```json
{
  "messages": [
    {"role": "user", "content": "A cinematic shot of a flying kite over the ocean."}
  ]
}
```

### Generation with Parameters

Use `extra_body` to pass generation parameters:

```json
{
  "messages": [
    {"role": "user", "content": "A cinematic shot of a flying kite over the ocean."}
  ],
  "extra_body": {
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    "fps": 24,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "guidance_scale_2": 4.0,
    "seed": 42,
    "negative_prompt": ""
  }
}
```

## Generation Parameters (extra_body)

| Parameter                | Type  | Default | Description                           |
| ------------------------ | ----- | ------- | ------------------------------------- |
| `height`                 | int   | None    | Video height in pixels                |
| `width`                  | int   | None    | Video width in pixels                 |
| `num_frames`             | int   | None    | Number of frames                      |
| `fps`                    | int   | 24      | Frames per second for exported MP4    |
| `num_inference_steps`    | int   | 40      | Number of denoising steps             |
| `guidance_scale`         | float | 4.0     | CFG guidance scale (low noise)        |
| `guidance_scale_2`        | float | 4.0     | CFG guidance scale (high noise)       |
| `seed`                   | int   | None    | Random seed (reproducible)            |
| `negative_prompt`        | str   | None    | Negative prompt                       |
| `num_outputs_per_prompt` | int   | 1       | Number of videos to generate          |

## Response Format

```json
{
  "id": "chatcmpl-xxx",
  "created": 1234567890,
  "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": [{
        "type": "video_url",
        "video_url": {
          "url": "data:video/mp4;base64,..."
        }
      }]
    },
    "finish_reason": "stop"
  }],
  "usage": {...}
}
```

## Extract Video

```bash
cat response.json | jq -r '.choices[0].message.content[0].video_url.url' | cut -d',' -f2 | base64 -d > output.mp4
```

## File Description

| File                        | Description            |
| --------------------------- | ---------------------- |
| `run_server.sh`             | Server startup script  |
| `run_curl_text_to_video.sh` | curl example           |
| `openai_chat_client.py`     | Python client          |
