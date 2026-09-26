# Stable Diffusion XL

> Text-to-image generation with Stability AI's SDXL Base 1.0

## Summary

- Vendor: Stability AI
- Model: `stabilityai/stable-diffusion-xl-base-1.0`
- Task: Text-to-image generation
- Mode: Offline inference / Online serving with the OpenAI-compatible images API
- Maintainer: Community

## When to use this recipe

Use this recipe when you want a high-quality, open-weight text-to-image model
that generates 1024x1024 images. SDXL Base 1.0 is one of the most widely used
diffusion models and produces photorealistic outputs with strong prompt
adherence. It requires ~6.5 GiB for model weights and runs inference at
~7.5 seconds per image (30 steps) on a single Intel Data Center GPU.

## References

- Model: <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>
- Architecture reference: <https://github.com/Stability-AI/generative-models>
- Related example under `examples/`:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)

## Hardware Support

## GPU

### 1x Intel Arc BMG GPU (XPU)

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: Intel XPU environment with Intel Arc BMG GPU
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

**Offline inference:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --prompt "a beautiful sunset over the ocean, photorealistic" \
  --height 1024 --width 1024 \
  --num-inference-steps 30 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output sdxl_output.png
```

**Online serving:**

```bash
vllm serve stabilityai/stable-diffusion-xl-base-1.0 --omni --port 8091
```

#### Verification

**For offline inference:** Check the output image file exists and has valid content.

**For online serving:**

```bash
curl -X POST http://localhost:8091/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "prompt": "a cute cat sitting on a windowsill, highly detailed, sharp focus",
    "negative_prompt": "blurry, out of focus, low quality",
    "guidance_scale": 7.5,
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

- Memory usage: Model weights ~6.5 GiB, peak memory ~10.9 GiB at 1024x1024 with CFG.
- Generation time: ~7.5 seconds for 30 inference steps at 1024x1024.
- `negative_prompt` requires `guidance_scale > 1` to take effect. Recommended range: 5.0-9.0.
- Default resolution is 1024x1024. Other supported sizes: 512x512, 768x768.
- The model uses dual CLIP text encoders (ViT-L/14 + OpenCLIP ViT-bigG/14) for rich text understanding.

### 1x NVIDIA GPU (24+ GB VRAM)

#### Environment

- OS: Linux
- Python: 3.10+
- Driver / runtime: NVIDIA CUDA environment with an A100
- vLLM version: Match the repository requirements for your checkout
- vLLM-Omni version or commit: Use the commit you are deploying from

#### Command

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --prompt "a beautiful sunset over the ocean, photorealistic" \
  --height 1024 --width 1024 \
  --num-inference-steps 30 \
  --guidance-scale 7.5 \
  --seed 42 \
  --output sdxl_output.png
```

#### Notes

- Memory usage: ~6.5 GiB model weights, ~10 GiB peak with CFG at 1024x1024.
- Fits on a single GPU with 24 GB VRAM (e.g., RTX 3090/4090, A5000).
- For GPUs with less VRAM, use `--enable-cpu-offload` to offload text encoders during diffusion.
