# Stable Diffusion 3.5

> Text-to-image serving with Stability AI's SD 3.5 family

## Summary

- Vendor: Stability AI
- Model: `stabilityai/stable-diffusion-3.5-medium`, `stabilityai/stable-diffusion-3.5-large`
- Task: Text-to-image generation
- Mode: Online serving with the OpenAI-compatible images API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a lightweight, fast text-to-image model that
fits on a single consumer or workstation GPU. SD 3.5-medium requires only
~17 GiB VRAM and generates 1024×1024 images in ~4 seconds, making it one of
the most accessible diffusion models in vLLM-Omni. SD 3.5-large produces
noticeably sharper images at the cost of ~34 GiB VRAM and ~21 seconds per
image.

## References

- Model (medium): <https://huggingface.co/stabilityai/stable-diffusion-3.5-medium>
- Model (large): <https://huggingface.co/stabilityai/stable-diffusion-3.5-large>
- Related example under `examples/`:
  [`examples/online_serving/text_to_image/README.md`](../../examples/online_serving/text_to_image/README.md)
- User guide:
  [`docs/user_guide/examples/online_serving/text_to_image.md`](../../docs/user_guide/examples/online_serving/text_to_image.md)

## Hardware Support

This recipe documents two CUDA GPU configurations on a single RTX A6000 48 GB,
one for each model variant. Extend it with more hardware sections as community
validation lands.

## GPU

### 1x NVIDIA RTX A6000 48GB — stable-diffusion-3.5-medium

#### Environment

- OS: Ubuntu 22.04
- Python: 3.12
- Driver / runtime: NVIDIA driver 580.126.09, CUDA 13.0
- vLLM version: 0.22.0
- vLLM-Omni version or commit: 0.22.0rc2.dev8+g41a795b5d

#### Command

```bash
vllm serve stabilityai/stable-diffusion-3.5-medium --omni --port 8091
```

#### Verification

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-3.5-medium",
    "prompt": "a cute cat sitting on a windowsill, highly detailed, sharp focus",
    "negative_prompt": "blurry, out of focus",
    "guidance_scale": 4.5,
    "n": 1,
    "size": "1024x1024"
  }' | python3 -c "
import json, base64, sys
data = json.load(sys.stdin)
with open('output.png', 'wb') as f:
    f.write(base64.b64decode(data['data'][0]['b64_json']))
print('saved output.png')
"
```

Or use the example client:

```bash
python examples/online_serving/text_to_image/openai_chat_client.py \
  --server http://localhost:8091 \
  --prompt "a cute cat sitting on a windowsill" \
  --output output.png
```

#### Notes

- Memory usage: Model weights ~15.6 GiB, peak VRAM ~17.0 GiB (no CFG), ~20.9 GiB with `guidance_scale > 1` (CFG enabled).
- Generation time: ~4 seconds without CFG, ~8 seconds with `guidance_scale=4.5`.
- Model loading time: ~55 seconds (including weight download on first run).
- `negative_prompt` requires `guidance_scale > 1` to take effect; SD3 defaults to 1.0 (no CFG).

### 1x NVIDIA RTX A6000 48GB — stable-diffusion-3.5-large

#### Environment

- OS: Ubuntu 22.04
- Python: 3.12
- Driver / runtime: NVIDIA driver 580.126.09, CUDA 13.0
- vLLM version: 0.22.0
- vLLM-Omni version or commit: 0.22.0rc2.dev8+g41a795b5d

#### Command

```bash
vllm serve stabilityai/stable-diffusion-3.5-large --omni --port 8091
```

> **Note**: `stable-diffusion-3.5-large` is a gated model. Accept the license
> on [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
> and set `HF_TOKEN` before running.

#### Verification

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-3.5-large",
    "prompt": "a cute cat sitting on a windowsill, highly detailed, sharp focus",
    "negative_prompt": "blurry, out of focus",
    "guidance_scale": 4.5,
    "n": 1,
    "size": "1024x1024"
  }' | python3 -c "
import json, base64, sys
data = json.load(sys.stdin)
with open('output.png', 'wb') as f:
    f.write(base64.b64decode(data['data'][0]['b64_json']))
print('saved output.png')
"
```

#### Notes

- Memory usage: Peak VRAM ~31.6 GiB at 1024×1024, ~45.5 GiB at 2048×2048 with `guidance_scale=4.5`.
- Generation time: ~19 seconds at 1024×1024, ~102 seconds at 2048×2048.
- Image quality is noticeably better than medium, with sharper details.
- `negative_prompt` requires `guidance_scale > 1` to take effect; SD3 defaults to 1.0 (no CFG).
- 2048×2048 fits within 48 GB but leaves little headroom (~45.5 GiB used out of 49 GiB).
