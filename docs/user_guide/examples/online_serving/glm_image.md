# GLM-Image Online Serving

GLM-Image is a 2-stage image generation model (AR + Diffusion) supported by vLLM-Omni's
declarative config system. The pipeline topology and stage structure are declared in
`vllm_omni/model_executor/models/glm_image/pipeline.py`; deployment knobs (GPU placement,
memory, sampling params) live in `vllm_omni/deploy/glm_image.yaml`.

## Start Server

```bash
vllm serve zai-org/GLM-Image --omni --port 8091
```

The config system auto-detects the pipeline from the model's `model_index.json` — no
manual `--stage-configs-path` or `--deploy-config` needed.

By default, stage 0 (AR) runs on GPU 0 and stage 1 (Diffusion) on GPU 1. To colocate
both stages on a single GPU, override per stage:

```bash
vllm serve zai-org/GLM-Image --omni --port 8091 \
    --stage-0-devices 0 --stage-1-devices 0
```

## API Calls

### Text-to-Image

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "A photorealistic mountain landscape at sunset"}
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "guidance_scale": 1.5,
      "seed": 42
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

### Image-to-Image (Image Editing)

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Convert this image to watercolor style"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,$(base64 -w0 input.png)}"}
        ]
      }
    ],
    "extra_body": {
      "height": 1024,
      "width": 1024,
      "num_inference_steps": 50,
      "guidance_scale": 1.5
    }
  }' | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

### Using the OpenAI Python SDK

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.chat.completions.create(
    model="zai-org/GLM-Image",
    messages=[{"role": "user", "content": "A beautiful sunset over the ocean"}],
    extra_body={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 1.5,
        "seed": 42,
    },
)

img_url = response.choices[0].message.content[0].image_url.url
_, b64_data = img_url.split(",", 1)
with open("output.png", "wb") as f:
    f.write(base64.b64decode(b64_data))
```

For general-purpose request methods (curl, OpenAI SDK, Python `requests`), see
the [Text-to-Image](text_to_image.md) and [Image-to-Image](image_to_image.md)
guides.

## Generation Parameters

When using `/v1/chat/completions`, pass these inside `extra_body` in the curl
JSON, or via the `extra_body` keyword argument in the OpenAI Python SDK (see the
[Diffusion Chat API guide](../../../serving/diffusion_chat_api.md)).
When using the dedicated [`/v1/images/generations`](../../../serving/image_generation_api.md)
or [`/v1/images/edits`](../../../serving/image_edit_api.md) endpoints, pass
the supported generation controls as top-level fields directly. For image
dimensions and count, use `size` and `n` rather than `height` or `width`.

| Parameter             | Type  | Default | Description                         |
| --------------------- | ----- | ------- | ----------------------------------- |
| `height`              | int   | 1024    | Image height in pixels              |
| `width`               | int   | 1024    | Image width in pixels               |
| `num_inference_steps` | int   | 50      | Number of diffusion denoising steps |
| `guidance_scale`      | float | 1.5     | Classifier-free guidance scale      |
| `seed`                | int   | None    | Optional random seed                |
| `negative_prompt`     | str   | None    | Negative prompt                     |

## Response Format

```json
{
  "id": "chatcmpl-xxx",
  "created": 1234567890,
  "model": "zai-org/GLM-Image",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": [
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,..."
            }
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {}
}
```

## Extract Image

```bash
cat response.json | jq -r '.choices[0].message.content[0].image_url.url' | cut -d',' -f2- | base64 -d > output.png
```

## Architecture

GLM-Image uses a 2-stage pipeline:

```
Stage 0 (AR Model)                Stage 1 (Diffusion)
┌───────────────────┐            ┌─────────────────────┐
│ vLLM-optimized    │  prior     │  GlmImagePipeline   │
│ GlmImageFor       │──tokens──►│  ┌───────────────┐  │
│ Conditional       │            │  │ DiT Denoiser  │  │
│ Generation        │            │  └───────┬───────┘  │
│ (9B AR model)     │            │          ▼          │
└───────────────────┘            │  ┌───────────────┐  │
        ▲                        │  │  VAE Decode   │──┼──► Image
        │                        │  └───────────────┘  │
   Text / Image                  └─────────────────────┘
     Input
```

## VRAM Requirements

| Stage             | VRAM                   |
| :---------------- | :--------------------- |
| Stage-0 (AR)      | **~18 GiB + KV Cache** |
| Stage-1 (DiT+VAE) | **~20 GiB**            |
| Total             | **~38 GiB + KV Cache** |

## FAQ

- If you encounter OOM errors, adjust `gpu_memory_utilization` in the deploy config:

```yaml
# In vllm_omni/deploy/glm_image.yaml, reduce from default 0.6:
gpu_memory_utilization: 0.5
```

- The first request may be slow due to model warmup. Subsequent requests will be faster.
