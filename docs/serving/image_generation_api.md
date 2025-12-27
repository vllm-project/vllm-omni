# Image Generation API

vLLM-Omni provides an OpenAI DALL-E compatible API for text-to-image generation using diffusion models.

Each server instance runs a single model (specified at startup via `vllm serve <model> --omni`).

## Quick Start

### Start the Server

For example...

```bash
# Qwen-Image
vllm serve Qwen/Qwen-Image --omni --port 8000

# Z-Image Turbo
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8000
```

### Generate Images

**Using curl:**

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a dragon laying over the spine of the Green Mountains of Vermont",
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
        "prompt": "a black and white cat wearing a princess tiara",
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
    prompt="a horse jumping over a fence nearby a babbling brook",
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
| `model` | string | server's model | Model to use (optional, should match server if specified) |
| `n` | integer | 1 | Number of images to generate (1-10) |
| `size` | string | model defaults | Image dimensions in WxH format (e.g., "1024x1024", "512x512") |
| `response_format` | string | "b64_json" | Response format (only "b64_json" supported) |
| `user` | string | null | User identifier for tracking |

#### vllm-omni Extension Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `negative_prompt` | string | null | Text describing what to avoid in the image |
| `num_inference_steps` | integer | model defaults | Number of diffusion steps |
| `guidance_scale` | float | model defaults | Classifier-free guidance scale (typically 0.0-20.0) |
| `true_cfg_scale` | float | model defaults | True CFG scale (model-specific parameter, may be ignored if not supported) |
| `seed` | integer | null | Random seed for reproducibility |

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
    "prompt": "a steampunk city set in a valley of the Adirondack mountains",
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
        "prompt": "a portrait of a skier in deep powder snow",
        "negative_prompt": "blurry, low quality, distorted, ugly",
        "num_inference_steps": 100,
        "size": "1024x1024",
    }
)
```

## Parameter Handling

The API passes parameters directly to the diffusion pipeline without model-specific transformation:

- **Default values**: When parameters are not specified, the underlying model uses its own defaults
- **Pass-through design**: User-provided values are forwarded directly to the diffusion engine
- **Minimal validation**: Only basic type checking and range validation at the API level

### Parameter Compatibility

The API passes parameters directly to the diffusion pipeline without model-specific validation.

- Unsupported parameters may be silently ignored by the model
- Incompatible values will result in errors from the underlying pipeline
- Recommended values vary by model - consult model documentation

**Best Practice:** Start with the model's recommended parameters, then adjust based on your needs.

## Error Responses

### 400 Bad Request

Invalid parameters (e.g., model mismatch):

```json
{
  "detail": "Invalid size format: '1024x'. Expected format: 'WIDTHxHEIGHT' (e.g., '1024x1024')."
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

### Out of Memory

If you encounter OOM errors:
1. Reduce image size: `"size": "512x512"`
2. Reduce inference steps: `"num_inference_steps": 25`
3. Generate fewer images: `"n": 1`

## Testing

Run the test suite to verify functionality:

```bash
# All image generation tests
pytest tests/entrypoints/openai_api/test_image_server.py -v

# Specific test
pytest tests/entrypoints/openai_api/test_image_server.py::test_generate_single_image -v
```

## Development

Enable debug logging to see prompts and generation details:

```bash
vllm serve Qwen/Qwen-Image --omni \
  --uvicorn-log-level debug
```
