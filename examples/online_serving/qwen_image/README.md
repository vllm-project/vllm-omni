# Qwen-Image Online Serving

This example demonstrates how to deploy Qwen-Image model for online image generation service using vLLM-Omni.

## Start Server

### Basic Start

```bash
vllm serve Qwen/Qwen-Image --omni --port 8091
```

### Start with Parameters

```bash
vllm serve Qwen/Qwen-Image --omni \
    --port 8091 \
    --num-gpus 1 \
    --num-inference-steps 50 \
    --guidance-scale 7.5 \
    --diffusion-seed 42
```

Or use the startup script:

```bash
bash run_server.sh
```

## API Calls

### Method 1: Using curl

```bash
# Basic text-to-image generation
bash run_curl_text_to_image.sh

# Or execute directly
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A beautiful landscape painting"}
    ]
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > output.png
```

### Method 2: Using Python Client

```bash
python openai_chat_client.py --prompt "A beautiful landscape painting" --output output.png
```

### Method 3: Using Gradio Demo

```bash
python gradio_demo.py
# Visit http://localhost:7860
```

## Request Format

### Simple Text Generation

```json
{
  "messages": [
    {"role": "user", "content": "A beautiful landscape painting"}
  ]
}
```

### Generation with Parameters

```json
{
  "messages": [
    {"role": "system", "content": "size=1024x1024 steps=50 guidance=7.5 seed=42"},
    {"role": "user", "content": "A beautiful landscape painting"}
  ]
}
```

### Multimodal Input (Text + Structured Content)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "A beautiful landscape painting"}
      ]
    }
  ]
}
```

## System Message Parameters

| Parameter | Format | Description |
|-----------|--------|-------------|
| `size` | `1024x1024` | Image size (width x height) |
| `steps` | `50` | Number of inference steps |
| `guidance` | `7.5` | CFG guidance scale |
| `seed` | `42` | Random seed (reproducible) |
| `negative` | `text` | Negative prompt |

## Response Format

```json
{
  "id": "chatcmpl-xxx",
  "created": 1234567890,
  "model": "Qwen/Qwen-Image",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": [{
        "type": "image_url",
        "image_url": {
          "url": "data:image/png;base64,..."
        }
      }]
    },
    "finish_reason": "stop"
  }],
  "usage": {...}
}
```

## Extract Image

```bash
# Extract base64 from response and decode to image
cat response.json | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > output.png
```

## File Description

| File | Description |
|------|-------------|
| `run_server.sh` | Server startup script |
| `run_curl_text_to_image.sh` | curl example |
| `openai_chat_client.py` | Python client |
| `gradio_demo.py` | Gradio interactive interface |
