# LTX-2.3 Text-to-Video with Audio on 1x GPU (96GB VRAM)

> 22B parameter text-to-video + audio generation model served via vLLM-Omni

## Summary

- Vendor: Lightricks
- Model: `dg845/LTX-2.3-Diffusers`
- Task: Text-to-video with synchronized audio generation
- Mode: Online serving (pure diffusion)
- Maintainer: @oglok

## When to use this recipe

Use this recipe when you want to serve LTX-2.3 for text-to-video generation
with audio. The model generates videos up to 20+ seconds at 768x512 resolution
with 48kHz audio, all from a single text prompt. Requires a GPU with at least
96GB VRAM due to the 22B parameter transformer (~44GB weights) plus text
encoder, VAE, and vocoder components.

## References

- Model: <https://huggingface.co/dg845/LTX-2.3-Diffusers>
- Requires `diffusers >= 0.38.0` (install from git: `pip install git+https://github.com/huggingface/diffusers.git`)

## Serving

### Command

```bash
vllm serve dg845/LTX-2.3-Diffusers \
  --omni \
  --model-class-name LTX23Pipeline \
  --stage-init-timeout 600
```

### Verification

```bash
# Health check
curl http://localhost:8000/health

# Generate a 3-second video (81 frames at 24fps)
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A majestic bald eagle soaring over a misty mountain valley at dawn, golden sunlight breaking through clouds" \
  -F "negative_prompt=blurry, low quality, distorted, watermark" \
  -F "model=dg845/LTX-2.3-Diffusers" \
  -F "num_frames=81" \
  -F "fps=24" \
  -F "size=768x512" \
  -F "num_inference_steps=30" \
  -F "guidance_scale=4.0" \
  -F "seed=42"

# Generate a 10-second video (241 frames)
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cozy Japanese ramen shop at night in the rain, steam rising from bowls, neon signs reflecting on wet cobblestone streets" \
  -F "model=dg845/LTX-2.3-Diffusers" \
  -F "num_frames=241" \
  -F "fps=24" \
  -F "size=768x512" \
  -F "num_inference_steps=30" \
  -F "guidance_scale=4.0"

# Generate a 20-second video (481 frames)
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=An underwater coral reef teeming with tropical fish, sea turtles gliding gracefully, National Geographic documentary style" \
  -F "model=dg845/LTX-2.3-Diffusers" \
  -F "num_frames=481" \
  -F "fps=24" \
  -F "size=768x512" \
  -F "num_inference_steps=30" \
  -F "guidance_scale=4.0"
```

### Notes

- Memory usage: Model loads at ~36 GiB, peaks at ~62 GiB during inference
- Key flags:
  - `--stage-init-timeout 600`: Required for the initial `torch.compile` warmup (~90-140 seconds on first request)
  - `--model-class-name LTX23Pipeline`: Selects the LTX-2.3 pipeline (not LTX-2)
- Audio: 48kHz AAC via BWE vocoder, automatically synced with video
- CPU offloading: Text encoder (Gemma-3-12B), connectors, VAE, audio VAE, and vocoder stay on CPU and are moved to GPU only when needed
- Supported resolutions: 768x512, 512x384 (must be divisible by 32)
- Frame rate: 24 fps
- Duration: Controlled by `num_frames` (frames = duration_seconds * 24 + 1)
- Known limitations:
  - No image-to-video support yet (LTX23ImageToVideoPipeline is a placeholder)
  - No CFG-parallel support (single-GPU only)
  - Requires `diffusers >= 0.38.0` (not yet on PyPI, install from git)

## Hardware Support

## GPU

### 1x NVIDIA RTX PRO 6000 Blackwell (96GB)

#### Environment

- OS: Ubuntu 22.04
- Python: 3.10+
- Driver / runtime: CUDA 13.0, Driver 580.126.09
- vLLM version: 0.19.x
- vLLM-Omni version: 0.19.x

### Validated configurations

| Duration | Frames | Resolution | Steps | Guidance | Inference Time | Peak VRAM |
|----------|--------|------------|-------|----------|----------------|-----------|
| 3s | 81 | 768x512 | 30 | 4.0 | ~110s | ~62 GB |
| 10s | 241 | 768x512 | 30 | 4.0 | ~130s | ~62 GB |
| 20s | 481 | 768x512 | 30 | 4.0 | ~420s | ~62 GB |
