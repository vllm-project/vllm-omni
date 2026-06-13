# Ideogram4 Online Serving

Start an OpenAI-compatible image generation server with Ideogram4. This implementation is adapted from the diffusers Ideogram4 pipeline and the upstream Ideogram AI implementation.

## Start Server

### Basic Start

```bash
vllm serve ideogram-ai/ideogram-4-nf4 --omni --port 8091
```

If you want the fp8 checkpoint, swap the model id:

```bash
vllm serve ideogram-ai/ideogram-4-fp8 --omni --port 8091
```

## API Calls

### Method 1: Using curl

```bash
curl -s http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A detailed character design sheet for a young explorer-mechanic-inventor in a frozen world.",
    "size": "1024x1024",
    "num_inference_steps": 50,
    "guidance_scale": 7.0,
    "seed": 42
  }' | jq -r '.data[0].b64_json' | base64 -d > ideogram4.png
```

### Method 2: Using OpenAI Python SDK

```python
from openai import OpenAI
import base64

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.images.generate(
    model="ideogram-ai/ideogram-4-nf4",
    prompt="A detailed character design sheet for a young explorer-mechanic-inventor in a frozen world.",
    n=1,
    size="1024x1024",
    response_format="b64_json",
)

img_url = response.data[0].b64_json
with open("ideogram4.png", "wb") as f:
    f.write(base64.b64decode(img_url))
```

## Request Parameters

- `prompt`: text description of the desired image.
- `size`: use `WIDTHxHEIGHT` (for example, `1024x1024`).
- `num_inference_steps`: diffusion steps; Ideogram4 defaults to 50 when omitted.
- `guidance_scale`: classifier-free guidance scale.
- `seed`: random seed for reproducibility.

## Notes

- Ideogram4 works best with structured JSON captions, but plain text prompts are also accepted.
- Accept the Hugging Face license gate before using `ideogram-ai/ideogram-4-nf4` or `ideogram-ai/ideogram-4-fp8`.
- Use `--port` to avoid clashing with other local services.
