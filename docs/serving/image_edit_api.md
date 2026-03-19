# Image Edit API

vLLM-Omni provides an OpenAI DALL-E compatible API for image edit using diffusion models.

Each server instance runs a single model (specified at startup via `vllm serve <model> --omni`).

## Quick Start

### Start the Server

For example...

```bash
# Qwen-Image
vllm serve Qwen/Qwen-Image-Edit-2511 --omni --port 8000


### Generate Images

**Using curl:**

```bash
curl -s -D >(grep -i x-request-id >&2) \
  -o >(jq -r '.data[0].b64_json' | base64 --decode > gift-basket.png) \
  -X POST "http://localhost:8000/v1/images/edits" \
  -F "model=xxx" \
  -F "image=@./xx.png" \
  -F "prompt='this bear is wearing sportwear. holding a basketball, and bending one leg.'" \
  -F "size=1024x1024" \
  -F "output_format=png"
```


**Using OpenAI SDK:**

```python
import base64
from openai import OpenAI
from pathlib import Path
client = OpenAI(
    api_key="None",
    base_url="http://localhost:8000/v1"
)

input_image_url = "https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png"

result = client.images.edit(
    image=[],
    model="Qwen-Image-Edit-2511",
    prompt="Change the bears in the two input images into walking together.",
    size='512x512',
    stream=False,
    output_format='jpeg',
    # url格式
    extra_body={
        "url": [input_image_url1,input_image_url],
        "num_inference_steps": 50,
        "guidance_scale": 1,
        "seed": 777,
    }
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("edit_out_http.jpeg", "wb") as f:
    f.write(image_bytes)
```

## API Reference

### Endpoint

```
POST /v1/images/edits
Content-Type: multipart/form-data
```

### Request Parameters

#### OpenAI Standard Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | A text description of the desired image |
| `model` | string | server's model | Model to use (optional, should match server if specified) |
| `image` | string or array | **required** | The image(s) to edit. |
| `n` | integer | 1 | Number of images to generate (1-10) |
| `size` | string | "auto" | Image dimensions in WxH format (e.g., "1024x1024", "512x512"), when set to auto, it decide size from first input image. |
| `response_format` | string | "b64_json" | Response format (only "b64_json" supported) |
| `user` | string | null | User identifier for tracking |
| `output_format` | string | "png" | The format in which the generated images are returned. Must be one of "png", "jpg", "jpeg", "webp". |
| `output_compression` | integer | 100 | The compression level (0-100%) for the generated images. |
| `background` | string or null | "auto" | Allows to set transparency for the background of the generated image(s).

#### vllm-omni Extension Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | string or array | None | The image(s) to edit. |
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
  ],
  "output_format": null,
  "size": null,
}
```

## Examples

### Multiple Images input

```bash
curl -s -D >(grep -i x-request-id >&2) \
  -o >(jq -r '.data[0].b64_json' | base64 --decode > gift-basket.png) \
  -X POST "http://localhost:8000/v1/images/edits" \
  -F "model=xxx" \
  -F "image=@xx.png" \
  -F "image=@xx.png"
  -F "prompt='this bear is wearing sportwear. holding a basketball, and bending one leg.'" \
  -F "size=1024x1024" \
  -F "output_format=png"
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
  "detail": "Field 'image' or 'url' is required"
}
```

## Troubleshooting

### Server Not Running

```bash
# Check if server is responding
curl -X http://localhost:8000/v1/images/edit \
  -F "prompt='test'"
```

### Out of Memory

If you encounter OOM errors:
1. Reduce image size: `"size": "512x512"`
2. Reduce inference steps: `"num_inference_steps": 25`

## Development

Enable debug logging to see prompts and generation details:

```bash
vllm serve Qwen/Qwen-Image-Edit-2511 --omni \
  --uvicorn-log-level debug
```
