# Image-To-Video

This example demonstrates how to deploy the Wan2.2 image-to-video model for online video generation using vLLM-Omni.

## Start Server

### Basic Start

```bash
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni --port 8091
```

### Start with Parameters

Or use the startup script:

```bash
bash run_server.sh
```

The script allows overriding:
- `MODEL` (default: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`)
- `PORT` (default: `8091`)
- `BOUNDARY_RATIO` (default: `0.875`)
- `FLOW_SHIFT` (default: `12.0`)
- `CACHE_BACKEND` (default: `none`)
- `ENABLE_CACHE_DIT_SUMMARY` (default: `0`)

## API Calls

### Method 1: Using curl

```bash
# Basic image-to-video generation
bash run_curl_image_to_video.sh

# Or execute directly (OpenAI-style multipart)
curl -X POST http://localhost:8091/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=A bear playing with yarn, smooth motion" \
  -F "negative_prompt=low quality, blurry, static" \
  -F "input_reference=@/path/to/qwen-bear.png" \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=33" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=1.0" \
  -F "guidance_scale_2=1.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=12.0" \
  -F "seed=42" | jq -r '.data[0].b64_json' | base64 -d > wan22_i2v_output.mp4
```

## Request Format

### Required Fields

```bash
curl -X POST http://localhost:8091/v1/videos \
  -F "prompt=A bear playing with yarn, smooth motion" \
  -F "negative_prompt=low quality, blurry, static" \
  -F "input_reference=@/path/to/qwen-bear.png"
```

### Generation with Parameters

```bash
curl -X POST http://localhost:8091/v1/videos \
  -F "prompt=A bear playing with yarn, smooth motion" \
  -F "negative_prompt=low quality, blurry, static" \
  -F "input_reference=@/path/to/qwen-bear.png" \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=33" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=1.0" \
  -F "guidance_scale_2=1.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=12.0" \
  -F "seed=42"
```
