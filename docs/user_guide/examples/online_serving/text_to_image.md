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

### Start with Parallelism Acceleration

Enable Tensor Parallelism and VAE Patch Parallelism for faster inference:

```bash
# With Tensor Parallelism (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --tensor-parallel-size 2

# With Tensor Parallelism and VAE Patch Parallelism (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --tensor-parallel-size 2 --vae-patch-parallel-size 2 --vae-use-tiling

# With Sequence Parallelism (Ulysses-SP, requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2

# With Ring-Attention (requires >= 2 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --ring 2

# Combined: Ulysses + Ring (requires >= 4 GPUs)
vllm serve Qwen/Qwen-Image --omni --port 8091 --usp 2 --ring 2
```

For more details on parallelism acceleration, see the [Parallelism Acceleration Guide](https://github.com/vllm-project/vllm-omni/tree/main/examples/diffusion/parallelism_acceleration.md).

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

### Method 2: Using OpenAI Python SDK

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.chat.completions.create(
    model="Qwen/Qwen-Image",
    messages=[{"role": "user", "content": "A beautiful landscape painting"}],
    extra_body={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "seed": 42,
    },
)

img_url = response.choices[0].message.content[0].image_url.url
_, b64_data = img_url.split(",", 1)
with open("output.png", "wb") as f:
    f.write(base64.b64decode(b64_data))
```

### Method 3: Using Python Client Script

```bash
python openai_chat_client.py --prompt "A beautiful landscape painting" --output output.png
```

### Method 4: Using Gradio Demo

```bash
python gradio_demo.py
# Visit http://localhost:7860
```

## LoRA

This example supports Peft-compatible LoRA (Low-Rank Adaptation) adapters for diffusion models. The LoRA adapter path must be readable on the **server** machine (usually a local path or a mounted directory).

### Using Python Client with LoRA

```bash
python openai_chat_client.py \
  --prompt "A piece of cheesecake" \
  --lora-path /path/to/lora_adapter \
  --lora-name my_lora \
  --lora-scale 1.0 \
  --output output.png
```

### Using curl with LoRA (Images API)

The `/v1/images/generations` endpoint supports a `lora` field in the request body:

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A piece of cheesecake",
    "size": "1024x1024",
    "seed": 42,
    "lora": {
      "name": "my_lora",
      "local_path": "/path/to/lora_adapter",
      "scale": 1.0
    }
  }' | jq -r '.data[0].b64_json' | base64 -d > output.png
```

### LoRA Parameters

| Parameter    | Type  | Description                                                               |
| ------------ | ----- | ------------------------------------------------------------------------- |
| `name`       | str   | LoRA adapter name (optional, defaults to path stem)                       |
| `local_path` | str   | Server-local path to LoRA adapter folder (PEFT format, required)          |
| `scale`      | float | LoRA scale factor (default: 1.0)                                          |
| `int_id`     | int   | LoRA integer ID for caching (optional, derived from path if not provided) |

### LoRA Adapter Format

LoRA adapters must be in PEFT (Parameter-Efficient Fine-Tuning) format. A typical LoRA adapter directory structure:

```
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
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

Wrap generation parameters inside `extra_body` in the request JSON:

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

!!! tip "Using the OpenAI SDK"
    When using the OpenAI Python SDK, pass these parameters via the `extra_body`
    keyword argument. The SDK merges them into the top-level request body automatically:

    ```python
    client.chat.completions.create(
        model="Qwen/Qwen-Image",
        messages=[...],
        extra_body={"height": 1024, "width": 1024, "num_inference_steps": 50},
    )
    ```

    For details on how generation parameters are handled across different clients, see the
    [Diffusion Chat API guide](../../../../serving/diffusion_chat_api.md).

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

## Generation Parameters

When using `/v1/chat/completions`, pass these inside `extra_body` in the curl
JSON, or via the `extra_body` keyword argument in the OpenAI Python SDK (see the
[Diffusion Chat API guide](../../../../serving/diffusion_chat_api.md)).
When using the dedicated [`/v1/images/generations`](../../../../serving/image_generation_api.md)
endpoint, pass the supported generation controls as top-level JSON fields
directly. For image dimensions and count, use `size` and `n` rather than
`height`, `width`, or `num_outputs_per_prompt`.

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
