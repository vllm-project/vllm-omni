# Text-To-Video

A unified script for text-to-video generation. Supports multiple models with model-aware defaults.

## Supported Models

| Model | Default Resolution | Default Frames | Default Steps | Guidance | VRAM (BF16) |
|---|---|---|---|---|---|
| `Wan-AI/Wan2.1-VACE-1.3B-diffusers` | 480x832 | 81 | 30 | 5.0 | ~20 GiB (RTX 5090, VAE tiling) |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 720x1280 | 81 | 40 | 4.0 | ~60 GiB |
| `diffusers/LTX-2.3-Diffusers` | 384x512 | 25 | 20 | 4.0 | 96GB-class GPU |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` | 480x832 | 121 | 50 | 6.0 | 1×A100 80GB |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | 720x1280 | 121 | 50 | 6.0 | FP8 + VAE tiling required |
| `nvidia/Cosmos3-Nano` | 720x1280 | 189 | 35 | 6.0 | ~46 GiB (peak, 720p) |
| `BestWishYsh/Helios-Base` / `Helios-Mid` / `Helios-Distilled` | 384x640 | 99 | 50 | 5.0 / 5.0 / 1.0 | — |

## Local CLI Usage

### Wan2.2 (default)

```bash
python text_to_video.py \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 33 \
  --guidance-scale 4.0 \
  --guidance-scale-high 3.0 \
  --flow-shift 12.0 \
  --num-inference-steps 40 \
  --fps 16 \
  --output t2v_out.mp4
```

### Wan2.1 VACE (T2V)

VACE text-to-video uses this shared entrypoint. Conditional VACE tasks use
the shared [`image_to_video.py`](../image_to_video/README.md#wan21-vace-conditional-tasks)
entrypoint, which constructs the pipeline-native conditioning data from the
provided media inputs. No explicit mode parameter is required.

```bash
python text_to_video.py \
  --model Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --prompt "A sleek, humanoid robot stands in a vast warehouse filled with neatly stacked cardboard boxes on industrial shelves." \
  --seed 0 \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --num-inference-steps 30 \
  --guidance-scale 5.0 \
  --flow-shift 5.0 \
  --vae-use-tiling \
  --output vace_t2v_output.mp4
```

LTX2 example:

```bash
python text_to_video.py \
  --model "Lightricks/LTX-2" \
  --prompt "A cinematic close-up of ocean waves at golden hour." \
  --negative-prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
  --height 512 \
  --width 768 \
  --num-frames 121 \
  --num-inference-steps 40 \
  --guidance-scale 4.0 \
  --frame-rate 24 \
  --output ltx2_out.mp4
```

### LTX-2.3

```bash
python text_to_video.py \
  --model diffusers/LTX-2.3-Diffusers \
  --model-class-name LTX23Pipeline \
  --prompt "Cherry blossoms swaying gently in the breeze with synchronized ambient sound" \
  --negative-prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
  --height 384 \
  --width 512 \
  --num-frames 25 \
  --num-inference-steps 20 \
  --guidance-scale 4.0 \
  --frame-rate 24 \
  --fps 24 \
  --audio-sample-rate 48000 \
  --output ltx23_t2v_output.mp4
```

Use the Diffusers-format checkpoint `diffusers/LTX-2.3-Diffusers`; the
upstream `Lightricks/LTX-2.3` raw safetensors repo is not directly loadable by
this pipeline. Pass `--model-class-name LTX23Pipeline` to select the LTX-2.3
text-to-video pipeline explicitly.

### HunyuanVideo-1.5 (480p)

```bash
python text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --prompt "A cat walks through a sunlit garden, flowers swaying gently in the breeze." \
  --height 480 \
  --width 832 \
  --num-frames 121 \
  --guidance-scale 6.0 \
  --flow-shift 5.0 \
  --num-inference-steps 50 \
  --fps 24 \
  --output hunyuan_video_15_output.mp4
```

### HunyuanVideo-1.5 (720p)

```bash
python text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --height 720 \
  --width 1280 \
  --num-frames 121 \
  --guidance-scale 6.0 \
  --flow-shift 9.0 \
  --num-inference-steps 50 \
  --fps 24 \
  --output hunyuan_720p.mp4
```

### HunyuanVideo-1.5 with FP8 Quantization

```bash
python text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --prompt "A dog running across a field of golden wheat." \
  --quantization fp8 \
  --height 480 --width 832 --num-frames 121 \
  --guidance-scale 6.0 --flow-shift 5.0 \
  --output hunyuan_fp8.mp4
```

Quick test (smaller resolution, fewer frames):

```bash
python text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --height 320 --width 576 --num-frames 17 --num-inference-steps 30 \
  --flow-shift 5.0 \
  --output quick_test.mp4
```

### Cosmos3

```bash
python text_to_video.py \
  --model nvidia/Cosmos3-Nano \
  --prompt "A robot arm is cleaning a plate in the kitchen." \
  --negative-prompt "blurry, distorted, low quality, jittery, deformed" \
  --height 720 --width 1280 --num-frames 189 --fps 24 \
  --num-inference-steps 35 --guidance-scale 6.0 \
  --extra-body '{"flow_shift": 10.0, "max_sequence_length": 4096, "guardrails": false,
                 "use_resolution_template": false, "use_duration_template": false}' \
  --output cosmos3_t2v.mp4
```

### Helios (T2V)

Helios ships three variants. Model-specific knobs (declared in
`vllm_omni/model_extras/helios.py`) are passed via the generic `--extra-body`
JSON flag rather than bespoke per-model flags.

**Helios-Base** (Stage 1 only):

```bash
python text_to_video.py \
  --model BestWishYsh/Helios-Base \
  --prompt "A dynamic time-lapse of scenery rushing past the window of a speeding train." \
  --guidance-scale 5.0 \
  --output helios_t2v_base.mp4
```

**Helios-Mid** (Stage 2 pyramid + CFG-Zero*):

```bash
python text_to_video.py \
  --model BestWishYsh/Helios-Mid \
  --prompt "A dynamic time-lapse of scenery rushing past the window of a speeding train." \
  --guidance-scale 5.0 \
  --extra-body '{"is_enable_stage2": true, "pyramid_num_inference_steps_list": [20, 20, 20], "use_cfg_zero_star": true, "use_zero_init": true, "zero_steps": 1}' \
  --output helios_t2v_mid.mp4
```

**Helios-Distilled** (Stage 2 pyramid + DMD, few-step):

```bash
python text_to_video.py \
  --model BestWishYsh/Helios-Distilled \
  --prompt "A dynamic time-lapse of scenery rushing past the window of a speeding train." \
  --num-frames 240 \
  --guidance-scale 1.0 \
  --extra-body '{"is_enable_stage2": true, "pyramid_num_inference_steps_list": [2, 2, 2], "is_amplify_first_chunk": true}' \
  --output helios_t2v_distilled.mp4
```

> Helios image-to-video (I2V) and video-to-video (V2V) require image/video
> conditioning tensors that cannot be passed through the JSON `--extra-body`
> flag; they are out of scope for this text-to-video example.

## Key Arguments

### Common

- `--model`: Diffusers model ID or local path.
- `--model-class-name`: Optional explicit pipeline class. Use `LTX23Pipeline`
  for LTX-2.3 text-to-video.
- `--prompt`: text description (string).
- `--height/--width`: output resolution. Default depends on model.
- `--num-frames`: number of frames. Default depends on model.
- `--guidance-scale`: CFG scale. Default depends on model.
- `--num-inference-steps`: sampling steps. Default depends on model.
- `--fps`: frames per second for the saved MP4.
- `--output`: path to save the generated video.
- `--extra-body`: JSON dict of model-specific knobs (declared in `vllm_omni/model_extras/`), merged into sampling `extra_args`. See the Helios recipes above.
- `--vae-use-slicing`: enable VAE slicing for memory optimization.
- `--vae-use-tiling`: enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](../../../docs/user_guide/diffusion/parallelism_acceleration.md#cfg-parallel).
- `--tensor-parallel-size`: tensor parallel size (effective for models that support TP, e.g. LTX2).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.
- `--enable-layerwise-offload`: enable layerwise offloading on DiT modules.
- `--frame-rate`: generation FPS for pipelines that require it (e.g., LTX2).
- `--audio-sample-rate`: audio sample rate for embedded audio (when the
  pipeline returns audio; LTX-2.3 outputs 48kHz audio).
- `--quantization`: quantization method (such as `fp8` for FP8).
- `--flow-shift`: scheduler flow_shift parameter.
- `--extra-body`: JSON object of model-specific generation params, filtered against the model's declared `extra_body_params` (see [`vllm_omni/model_extras`](../../../vllm_omni/model_extras)). Used by Cosmos3 (see above).

### Wan2.2-specific

- `--negative-prompt`: artifacts to suppress.
- `--guidance-scale-high`: separate CFG scale for high-noise stage.
- `--boundary-ratio`: boundary split for low/high DiT (default 0.875).
- `--flow-shift`: scheduler flow_shift (5.0 for 720p, 12.0 for 480p).
- `--cache-backend`: `cache_dit` for acceleration.

### HunyuanVideo-1.5 Optimal Configs

| Variant | flow_shift | guidance_scale | steps |
|---------|-----------|----------------|-------|
| 480p T2V | 5.0 | 6.0 | 50 |
| 720p T2V | 9.0 | 6.0 | 50 |
| 480p I2V | 5.0 | 6.0 | 50 |
| 720p I2V | 7.0 | 6.0 | 50 |
| CFG-distilled | (same) | 1.0 | 50 |

> If you encounter OOM errors, try `--vae-use-slicing`, `--vae-use-tiling`, `--enable-cpu-offload`, or `--quantization fp8`.
