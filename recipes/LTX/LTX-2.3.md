# LTX-2.3

> 22B parameter text-to-video + audio generation model served via vLLM-Omni

## Summary

- Vendor: Lightricks
- Model: `dg845/LTX-2.3-Diffusers`
- Task: Text-to-video and image-to-video with synchronized audio generation
- Mode: Online serving (pure diffusion)
- Maintainer: @oglok

## When to use this recipe

Use this recipe when you want to serve LTX-2.3 for text-to-video or
image-to-video generation with audio. The model generates videos up to 20+
seconds at 768x512 resolution with 48kHz audio from text prompts, and can
condition video generation on an initial image. Start with a 96GB-class GPU for
validation because the model combines a 22B parameter transformer with text
encoder, VAE, and vocoder components.

## References

- Upstream raw checkpoints: <https://huggingface.co/Lightricks/LTX-2.3>
- Diffusers-format checkpoint: <https://huggingface.co/dg845/LTX-2.3-Diffusers>
- Requires `diffusers >= 0.38.0` (install from git: `pip install git+https://github.com/huggingface/diffusers.git`)

## Serving

### Command

```bash
vllm serve dg845/LTX-2.3-Diffusers \
  --omni \
  --model-class-name LTX23Pipeline \
  --stage-init-timeout 600
```

For image-to-video:

```bash
vllm serve dg845/LTX-2.3-Diffusers \
  --omni \
  --model-class-name LTX23ImageToVideoPipeline \
  --stage-init-timeout 600
```

### T2V Verification

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
  -F "seed=42"

# Generate a 10-second video (241 frames)
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cozy Japanese ramen shop at night in the rain, steam rising from bowls, neon signs reflecting on wet cobblestone streets" \
  -F "model=dg845/LTX-2.3-Diffusers" \
  -F "num_frames=241" \
  -F "fps=24" \
  -F "size=768x512" \
  -F "num_inference_steps=30"

# Generate a 20-second video (481 frames)
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=An underwater coral reef teeming with tropical fish, sea turtles gliding gracefully, National Geographic documentary style" \
  -F "model=dg845/LTX-2.3-Diffusers" \
  -F "num_frames=481" \
  -F "fps=24" \
  -F "size=768x512" \
  -F "num_inference_steps=30"
```

### I2V Verification

Start the server with `--model-class-name LTX23ImageToVideoPipeline`, then pass
an image reference. The I2V pipeline requires `input_reference` or
`image_reference` because the first frame is the conditioning input.

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/videos/sync \
  -F "prompt=A plush toy astronaut gently waving while the camera slowly pushes in." \
  -F "input_reference=@/absolute/path/to/reference.png" \
  -F "model=dg845/LTX-2.3-Diffusers" \
  -F "num_frames=81" \
  -F "fps=24" \
  -F "size=768x512" \
  -F "num_inference_steps=30" \
  -F "seed=42" \
  -o ltx23_i2v.mp4
```

### Guidance Controls

`LTX23Pipeline` defaults to the official one-stage non-HQ guidance recipe:
video CFG `3.0`, audio CFG `7.0`, STG `1.0`, modality guidance `3.0`,
rescale `0.7`, and STG block `[28]`.

Use `extra_params` for LTX-2.3-specific controls:

| Control | Official default | Disable / neutral value |
|---|---:|---:|
| `video_cfg_scale` / `audio_cfg_scale` | `3.0` / `7.0` | `1.0` |
| `video_stg_scale` / `audio_stg_scale` | `1.0` | `0.0` |
| `video_modality_scale` / `audio_modality_scale` | `3.0` | `1.0` |
| `video_rescale_scale` / `audio_rescale_scale` | `0.7` | `0.0` |
| `video_stg_blocks` / `audio_stg_blocks` | `[28]` | `[]` |

`guidance_scale` is only the generic CFG fallback. It does not disable STG,
modality guidance, or rescale, and specific `extra_params` values take
precedence. Partial overrides keep the remaining official defaults.

To run CFG-only, or to use CFG parallelism, set matching video/audio CFG scales
and disable the other guidance branches:

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A serene lakeside sunrise with mist over the water." \
  -F "model=dg845/LTX-2.3-Diffusers" \
  -F "num_frames=25" \
  -F "fps=24" \
  -F "size=512x384" \
  -F "num_inference_steps=20" \
  -F "guidance_scale=4.0" \
  -F 'extra_params={"audio_cfg_scale": 4.0, "video_stg_scale": 0.0, "audio_stg_scale": 0.0, "video_modality_scale": 1.0, "audio_modality_scale": 1.0, "video_rescale_scale": 0.0, "audio_rescale_scale": 0.0, "video_stg_blocks": [], "audio_stg_blocks": []}'
```

### Notes

- Memory usage: run a latest-head serving smoke or DFX benchmark on the target
  GPU before publishing load/peak VRAM numbers.
- Checkpoint format: use a Diffusers-format checkpoint such as
  `dg845/LTX-2.3-Diffusers`; the upstream `Lightricks/LTX-2.3` repository ships
  raw safetensors and does not contain the subfolder configs required by this
  pipeline loader.
- Key flags:
  - `--stage-init-timeout 600`: Gives the initial `torch.compile` warmup enough time on large checkpoints
  - `--model-class-name LTX23Pipeline`: Selects the LTX-2.3 text-to-video pipeline (not LTX-2)
  - `--model-class-name LTX23ImageToVideoPipeline`: Selects the LTX-2.3 image-to-video pipeline
  - `input_reference=@...`: Uploads the I2V reference image directly
- Audio: 48kHz AAC via BWE vocoder, automatically synced with video
- CPU offloading: Text encoder (Gemma-3-12B), connectors, VAE, audio VAE, and vocoder stay on CPU and are moved to GPU only when needed
- Supported resolutions: 768x512, 512x384 (must be divisible by 32)
- Frame rate: 24 fps
- Duration: Controlled by `num_frames` (frames = duration_seconds * 24 + 1)
- Known limitations:
  - LTX-2.3 I2V currently supports first-frame image conditioning, matching the LTX image-conditioning contract.
  - Requires `diffusers >= 0.38.0` (not yet on PyPI, install from git)

## Hardware Support

## GPU

### 1x NVIDIA RTX PRO 6000 Blackwell (96GB)

#### Environment

- OS: Ubuntu 22.04
- Python: 3.10+
- Driver / runtime: CUDA 13.0, Driver 580.126.09
- vLLM version: Match the PR or release checkout being validated
- vLLM-Omni version or commit: Use the commit you are deploying from

### Benchmarking

Do not copy latency or VRAM numbers between commits or machines. Record them
only after running a latest-head sweep on the target hardware.

For formal PR benchmarking, add or update a checked-in DFX perf config in
`tests/dfx/perf/tests/`, then run it with
`tests/dfx/perf/scripts/run_diffusion_benchmark.py`. At minimum, include a
single-device eager baseline for a small 384x512, 25-frame, 20-step workload.
If the PR changes parallel execution, include the corresponding CFG/TP/USP case
in the same config so the result is comparable on the target hardware.
