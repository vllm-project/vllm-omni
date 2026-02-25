# Image-To-Image

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/image_to_image>.


This example demonstrates how to deploy Qwen-Image-Edit model for online image editing service using vLLM-Omni.

For **multi-image** input editing, use **Qwen-Image-Edit-2509** (QwenImageEditPlusPipeline) and send multiple images in the user message content.

## Start Server

### Basic Start

```bash
vllm serve Qwen/Qwen-Image-Edit --omni --port 8092
```

!!! note
    If you encounter Out-of-Memory (OOM) issues or have limited GPU memory, you can enable VAE slicing and tiling to reduce memory usage, --vae-use-slicing --vae-use-tiling

### Multi-Image Edit (Qwen-Image-Edit-2509)

```bash
vllm serve Qwen/Qwen-Image-Edit-2509 --omni --port 8092
```

### Start with Parameters


Or use the startup script:

```bash
bash run_server.sh
```

To serve Qwen-Image-Edit-2509 with the script:

```bash
MODEL=Qwen/Qwen-Image-Edit-2509 bash run_server.sh
```

## API Calls

### Method 1: Using curl (Image Editing)

```bash
# Image editing
bash run_curl_image_edit.sh input.png "Convert this image to watercolor style"

# Or execute directly
IMG_B64=$(base64 -w0 input.png)

cat <<EOF > request.json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Convert this image to watercolor style"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,$IMG_B64"}}
    ]
  }],
  "extra_body": {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 1,
    "seed": 42
  }
}
EOF

curl -s http://localhost:8092/v1/chat/completions   -H "Content-Type: application/json"   -d @request.json | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2 | base64 -d > output.png
```

### Method 2: Using Python Client

```bash
python openai_chat_client.py --input input.png --prompt "Convert to oil painting style" --output output.png

# Multi-image editing (Qwen-Image-Edit-2509 server required)
python openai_chat_client.py --input input1.png input2.png --prompt "Combine these images into a single scene" --output output.png
```

### Method 3: Using Gradio Demo

```bash
python gradio_demo.py
# Visit http://localhost:7861
```

## Request Format

### Image Editing (Using image_url Format)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Convert this image to watercolor style"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ]
}
```

### Image Editing (Using Simplified image Format)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"text": "Convert this image to watercolor style"},
        {"image": "BASE64_IMAGE_DATA"}
      ]
    }
  ]
}
```

### Image Editing with Parameters

Use `extra_body` to pass generation parameters:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Convert to ink wash painting style"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
      ]
    }
  ],
  "extra_body": {
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "seed": 42
  }
}
```

### Multi-Image Editing (Qwen-Image-Edit-2509)

Provide multiple images in `content` (order matters):

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Combine these images into a single scene"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."} },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."} }
      ]
    }
  ]
}
```

## Generation Parameters (extra_body)

| Parameter                | Type  | Default | Description                           |
| ------------------------ | ----- | ------- | ------------------------------------- |
| `height`                 | int   | None    | Output image height in pixels         |
| `width`                  | int   | None    | Output image width in pixels          |
| `size`                   | str   | None    | Output image size (e.g., "1024x1024") |
| `num_inference_steps`    | int   | 50      | Number of denoising steps             |
| `guidance_scale`         | float | 7.5     | CFG guidance scale                    |
| `seed`                   | int   | None    | Random seed (reproducible)            |
| `negative_prompt`        | str   | None    | Negative prompt                       |
| `num_outputs_per_prompt` | int   | 1       | Number of images to generate          |

## Response Format

```json
{
  "id": "chatcmpl-xxx",
  "created": 1234567890,
  "model": "Qwen/Qwen-Image-Edit",
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

## Common Editing Instructions Examples

| Instruction                              | Description      |
| ---------------------------------------- | ---------------- |
| `Convert this image to watercolor style` | Style transfer   |
| `Convert the image to black and white`   | Desaturation     |
| `Enhance the color saturation`           | Color adjustment |
| `Convert to cartoon style`               | Cartoonization   |
| `Add vintage filter effect`              | Filter effect    |
| `Convert daytime scene to nighttime`     | Scene conversion |

## File Description

| File                     | Description                  |
| ------------------------ | ---------------------------- |
| `run_server.sh`          | Server startup script        |
| `run_curl_image_edit.sh` | curl image editing example   |
| `openai_chat_client.py`  | Python client                |
| `gradio_demo.py`         | Gradio interactive interface |

## Example materials

??? abstract "gradio_demo.py"
    ``````py
    --8<-- "examples/online_serving/image_to_image/gradio_demo.py"
    ``````
??? abstract "openai_chat_client.py"
    ``````py
    --8<-- "examples/online_serving/image_to_image/openai_chat_client.py"
    ``````
??? abstract "run_curl_image_edit.sh"
    ``````sh
    --8<-- "examples/online_serving/image_to_image/run_curl_image_edit.sh"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/image_to_image/run_server.sh"
    ``````
