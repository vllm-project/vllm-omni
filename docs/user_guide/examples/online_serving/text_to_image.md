# Text-To-Image

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/text_to_image>.


This example demonstrates how to deploy Qwen-Image model for online image generation service using vLLM-Omni.

## Start Server

### Basic Start

```bash
vllm serve Qwen/Qwen-Image --omni --port 8091
```
!!! note
    If you encounter Out-of-Memory (OOM) issues or have limited GPU memory, you can enable VAE slicing and tiling to reduce memory usage, --vae-use-slicing --vae-use-tiling

### Start with Parameters

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
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "true_cfg_scale": 4.0,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
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

Use `extra_body` to pass generation parameters:

```json
{
  "messages": [
    {"role": "user", "content": "A beautiful landscape painting"}
  ],
  "extra_body": {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "true_cfg_scale": 4.0,
    "seed": 42
  }
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

## Generation Parameters (extra_body)

| Parameter                | Type  | Default | Description                    |
| ------------------------ | ----- | ------- | ------------------------------ |
| `height`                 | int   | None    | Image height in pixels         |
| `width`                  | int   | None    | Image width in pixels          |
| `size`                   | str   | None    | Image size (e.g., "1024x1024") |
| `num_inference_steps`    | int   | 50      | Number of denoising steps      |
| `true_cfg_scale`         | float | 4.0     | Qwen-Image CFG scale           |
| `seed`                   | int   | None    | Random seed (reproducible)     |
| `negative_prompt`        | str   | None    | Negative prompt                |
| `num_outputs_per_prompt` | int   | 1       | Number of images to generate   |
| `--cfg-parallel-size`.   | int   | 1       | Number of GPUs for CFG parallelism |

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
cat response.json | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

## File Description

| File                        | Description                  |
| --------------------------- | ---------------------------- |
| `run_server.sh`             | Server startup script        |
| `run_curl_text_to_image.sh` | curl example                 |
| `openai_chat_client.py`     | Python client                |
| `gradio_demo.py`            | Gradio interactive interface |

## Example materials

??? abstract "gradio_demo.py"
    ``````py
    --8<-- "examples/online_serving/text_to_image/gradio_demo.py"
    ``````
??? abstract "openai_chat_client.py"
    ``````py
    --8<-- "examples/online_serving/text_to_image/openai_chat_client.py"
    ``````
??? abstract "run_curl_text_to_image.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_image/run_curl_text_to_image.sh"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_image/run_server.sh"
    ``````
