# Image Generation API

vLLM-Omni provides an OpenAI DALL-E compatible API for text-to-image generation using diffusion models.

## Supported Models

| Model | Default Steps | Max Steps | Guidance Scale | Notes |
|-------|---------------|-----------|----------------|-------|
| **Qwen/Qwen-Image** | 50 | 200 | 1.0 (configurable) | Supports true_cfg_scale=4.0 |
| **Tongyi-MAI/Z-Image-Turbo** | 9 | 16 | 0.0 (forced) | Fast generation, distilled for CFG=0 |

Each server instance runs a single model (specified at startup via `vllm serve <model> --omni`).

## Quick Start

### Start the Server

```bash
# Qwen-Image (full-featured)
vllm serve Qwen/Qwen-Image --omni --port 8000

# Z-Image Turbo (fast generation)
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8000
```

### Generate Images

**Using curl:**

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a dragon flying over mountains",
    "size": "1024x1024",
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > dragon.png
```

**Using Python:**

```python
import requests
import base64
from PIL import Image
import io

response = requests.post(
    "http://localhost:8000/v1/images/generations",
    json={
        "prompt": "a white siamese cat",
        "size": "1024x1024",
        "num_inference_steps": 50,
        "seed": 42,
    }
)

# Decode and save
img_data = response.json()["data"][0]["b64_json"]
img_bytes = base64.b64decode(img_data)
img = Image.open(io.BytesIO(img_bytes))
img.save("cat.png")
```

**Using OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

response = client.images.generate(
    model="Qwen/Qwen-Image",
    prompt="astronaut riding a horse",
    n=1,
    size="1024x1024",
    response_format="b64_json"
)

# Note: Extension parameters (seed, steps, cfg) require direct HTTP requests
```

## API Reference

### Endpoint

```
POST /v1/images/generations
Content-Type: application/json
```

### Request Parameters

#### OpenAI Standard Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text description of the desired image |
| `model` | string | server's model | Model to use (optional, must match server if specified) |
| `n` | integer | 1 | Number of images to generate (1-10) |
| `size` | string | "1024x1024" | Image dimensions in WxH format (e.g., "1024x1024", "512x512") |
| `response_format` | string | "b64_json" | Response format (only "b64_json" supported) |
| `user` | string | null | User identifier for tracking |

#### vllm-omni Extension Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `negative_prompt` | string | null | Text describing what to avoid in the image |
| `num_inference_steps` | integer | model-dependent | Number of diffusion steps (1-200, see model table) |
| `guidance_scale` | float | model-dependent | Classifier-free guidance scale (0.0-20.0) |
| `true_cfg_scale` | float | model-dependent | True CFG scale for Qwen-Image (ignored by Z-Image) |
| `seed` | integer | null | Random seed for reproducibility |

**Model-Specific Behavior:**
- **Z-Image Turbo**: `guidance_scale` is always forced to 0.0 (user input ignored), `true_cfg_scale` is ignored
- **Qwen-Image**: Uses `true_cfg_scale=4.0` by default, `guidance_scale=1.0` is configurable
- Exceeding a model's `max_steps` will result in a 400 error

### Response Format

```json
{
  "created": 1701234567,
  "data": [
    {
      "b64_json": "<base64-encoded PNG>",
      "url": null,
      "revised_prompt": null
    }
  ]
}
```

## Examples

### Multiple Images

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a steampunk city",
    "n": 4,
    "size": "1024x1024",
    "seed": 123
  }'
```

This generates 4 images in a single request.

### With Negative Prompt

```python
response = requests.post(
    "http://localhost:8000/v1/images/generations",
    json={
        "prompt": "beautiful mountain landscape",
        "negative_prompt": "blurry, low quality, distorted, ugly",
        "num_inference_steps": 100,
        "size": "1024x1024",
    }
)
```

### Z-Image Turbo (Fast Generation)

```bash
# Start Z-Image Turbo server
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8000

# Generate image with optimal settings
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "sunset over ocean waves",
    "num_inference_steps": 9,
    "size": "1024x1024"
  }' | jq -r '.data[0].b64_json' | base64 -d > sunset.png
```

**Note:** Z-Image Turbo is optimized for 9 steps and will reject requests with >16 steps.

## Model Profiles

vLLM-Omni uses **model profiles** to encapsulate model-specific behavior, parameters, and constraints. This abstraction allows the API to handle different diffusion models through a unified interface while respecting each model's unique characteristics.

**Key benefits:**
- Per-model default parameters (steps, guidance scales, dimensions)
- Automatic parameter validation and enforcement
- Easy addition of new models without changing API code
- Graceful handling of unsupported parameters

See `vllm_omni/entrypoints/openai/image_model_profiles.py` for profile definitions.

## Error Responses

### 400 Bad Request

Invalid parameters:

```json
{
  "detail": "num_inference_steps (20) exceeds model maximum (16)"
}
```

### 422 Unprocessable Entity

Validation errors (missing required fields):

```json
{
  "detail": [
    {
      "loc": ["body", "prompt"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 503 Service Unavailable

Diffusion engine not initialized:

```json
{
  "detail": "Diffusion engine not initialized. Start server with a diffusion model."
}
```

## Troubleshooting

### Server Not Running

```bash
# Check if server is responding
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "test"}'
```

### CUDA Out of Memory

If you encounter OOM errors:
1. Reduce image size: `"size": "512x512"`
2. Reduce inference steps: `"num_inference_steps": 25`
3. Generate fewer images: `"n": 1`

The server automatically enables VAE slicing and tiling for memory optimization.

### Invalid Size Format

Size must be in `WIDTHxHEIGHT` format (e.g., `"1024x1024"`, `"512x768"`). Common sizes:
- `256x256`
- `512x512`
- `1024x1024`
- `1792x1024`
- `1024x1792`

Custom sizes are supported but must be positive integers.

## Testing

Run the test suite to verify functionality:

```bash
# All image generation tests
pytest tests/entrypoints/openai/test_image_server.py -v

# Specific test
pytest tests/entrypoints/openai/test_image_server.py::test_generate_single_image -v
```

## Development

Enable debug logging to see prompts and generation details:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --uvicorn-log-level debug
```
