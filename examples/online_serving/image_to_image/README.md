# Image-To-Image

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

### Method 2: Using OpenAI Python SDK

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8092/v1", api_key="none")

with open("input.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen-Image-Edit",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Convert to watercolor style"},
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{img_b64}"
            }},
        ],
    }],
    extra_body={
        "num_inference_steps": 50,
        "guidance_scale": 1,
        "seed": 42,
    },
)

img_url = response.choices[0].message.content[0].image_url.url
_, b64_data = img_url.split(",", 1)
with open("output.png", "wb") as f:
    f.write(base64.b64decode(b64_data))
```

!!! note
    The OpenAI SDK's `extra_body` keyword argument merges parameters into the
    top-level request body automatically. When using curl or Python `requests`,
    wrap generation parameters inside a literal `"extra_body"` key in the JSON
    instead (as shown in the curl example above).

### Method 3: Using Python Client Script

```bash
python openai_chat_client.py --input input.png --prompt "Convert to oil painting style" --output output.png

# Multi-image editing (Qwen-Image-Edit-2509 server required)
python openai_chat_client.py --input input1.png input2.png --prompt "Combine these images into a single scene" --output output.png
```

### Method 4: Using Gradio Demo

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

### Layered Image Generation (Qwen-Image-Layered)

Qwen-Image-Layered generates multiple decomposed layers from a reference image and a text prompt.
Start the server with:

```bash
vllm serve Qwen/Qwen-Image-Layered --omni --port 8093
```

**Using curl**

```bash
IMG_B64=$(base64 -w0 input.png)

curl -sS http://localhost:8093/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg img "$IMG_B64" '{
    messages: [{
      role: "user",
      content: [
        {type: "image_url", image_url: {url: ("data:image/png;base64," + $img)}},
        {type: "text", text: "a rabbit"}
      ]
    }],
    extra_body: {
      num_inference_steps: 50,
      cfg_scale: 4.0,
      seed: 0,
      layers: 4,
      resolution: 640
    }
  }')" \
  | jq -r '.choices[0].message.content[] | .image_url.url | split(",")[1]' \
  | while IFS= read -r b64; do
      ((i++)); echo "$b64" | base64 -d > "layer_${i}.png"
    done
```

**Using Python**

```python
import base64
import requests

with open("input.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

payload = {
    "messages": [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/png;base64,{img_b64}"
            }},
            {"type": "text", "text": "a rabbit"},
        ],
    }],
    "extra_body": {
        "num_inference_steps": 50,
        "cfg_scale": 4.0,
        "seed": 0,
        "layers": 4,
        "resolution": 640,
    },
}

resp = requests.post(
    "http://localhost:8093/v1/chat/completions",
    json=payload,
    timeout=600,
)
data = resp.json()

for i, item in enumerate(data["choices"][0]["message"]["content"]):
    _, b64_data = item["image_url"]["url"].split(",", 1)
    with open(f"layer_{i}.png", "wb") as f:
        f.write(base64.b64decode(b64_data))
```

The response contains multiple images in `choices[0].message.content` — one per generated layer.

#### Qwen-Image-Layered Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layers` | int | 4 | Number of layers to decompose |
| `resolution` | int | 640 | Resolution for dimension calculation (640 or 1024) |
| `cfg_scale` | float | 4.0 | Classifier-free guidance scale (alias for `true_cfg_scale`) |
| `num_inference_steps` | int | 50 | Number of denoising steps |
| `seed` | int | None | Random seed for reproducibility |

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

## Generation Parameters

When using `/v1/chat/completions`, pass these inside `extra_body` in the curl
JSON, or via the `extra_body` keyword argument in the OpenAI Python SDK.
When using the dedicated `/v1/images/edits` endpoint, pass the supported
generation controls as top-level form fields directly. For image dimensions and
count, use `size` and `n` rather than `height`, `width`, or
`num_outputs_per_prompt`.

| Parameter                | Type  | Default | Description                           |
| ------------------------ | ----- | ------- | ------------------------------------- |
| `height`                 | int   | None    | Output image height in pixels         |
| `width`                  | int   | None    | Output image width in pixels          |
| `size`                   | str   | None    | Output image size (e.g., "1024x1024") |
| `num_inference_steps`    | int   | 50      | Number of denoising steps             |
| `guidance_scale`         | float | 1.0     | CFG guidance scale                    |
| `seed`                   | int   | None    | Random seed (reproducible)            |
| `negative_prompt`        | str   | None    | Negative prompt                       |
| `num_outputs_per_prompt` | int   | 1       | Number of images to generate          |
| `strength`               | float | 0.6     | **Z-Image only** - Denoising start timestep for I2I. Range: [0.0, 1.0]. Lower preserves more of original image. |
| `layers`                 | int   | 4       | Number of layers (Qwen-Image-Layered) |
| `resolution`             | int   | 640     | Resolution, 640 or 1024 (Qwen-Image-Layered) |

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
