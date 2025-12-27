# Text-To-Video

This example demonstrates how to deploy Wan2.2 models for online video generation using vLLM-Omni.

## Start Server

### Basic Start

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8093
```

### Start with Parameters

Or use the startup script:

```bash
bash run_server.sh
```

## API Calls

### Method 1: Using curl (Text-to-Video)

```bash
bash run_curl_text_to_video.sh
```

### Method 2: Using Python Client

```bash
python openai_chat_client.py --prompt "A cat surfing on waves" --output output.mp4
```

### Image-to-Video (Python Client)

```bash
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni --port 8093
python openai_chat_client.py --image input.jpg --prompt "A cat dancing" --output i2v.mp4
```

## Request Format

### Simple Text Generation

```json
{
  "messages": [
    {"role": "user", "content": "A cat surfing on waves"}
  ]
}
```

### Image-to-Video (Text + Image)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "A cat dancing"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}} 
      ]
    }
  ]
}
```

### Generation with Parameters

Use `extra_body` to pass generation parameters:

```json
{
  "messages": [
    {"role": "user", "content": "A cat surfing on waves"}
  ],
  "extra_body": {
    "height": 480,
    "width": 640,
    "num_frames": 32,
    "num_inference_steps": 40,
    "guidance_scale": 4.0,
    "guidance_scale_2": 3.0,
    "seed": 42,
    "fps": 16
  }
}
```

## Generation Parameters (extra_body)

| Parameter                | Type  | Default | Description                           |
| ------------------------ | ----- | ------- | ------------------------------------- |
| `height`                 | int   | None    | Video height in pixels                |
| `width`                  | int   | None    | Video width in pixels                 |
| `num_frames`             | int   | 81      | Number of frames                      |
| `num_inference_steps`    | int   | 40      | Number of denoising steps             |
| `guidance_scale`         | float | 4.0     | CFG scale (low/high if boundary set)  |
| `guidance_scale_2`       | float | None    | High-noise CFG scale                  |
| `seed`                   | int   | None    | Random seed                           |
| `negative_prompt`        | str   | None    | Negative prompt                       |
| `fps`                    | int   | 16      | Frames per second for MP4 export      |
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
  "usage": { ... }
}
```

## Extract Video

```bash
cat response.json | jq -r '.choices[0].message.content[0].video_url.url' | cut -d',' -f2 | base64 -d > output.mp4
```

## File Description

| File                        | Description           |
| --------------------------- | --------------------- |
| `run_server.sh`             | Server startup script |
| `run_curl_text_to_video.sh` | curl example          |
| `openai_chat_client.py`     | Python client         |
