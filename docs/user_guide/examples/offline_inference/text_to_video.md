# Text-To-Video

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_video>.


A unified script for text-to-video generation. Supports multiple models with model-aware defaults.

For backend selection and SageAttention usage, see the [Diffusion Attention Backends](../../diffusion/attention_backends.md) guide.

## Supported Models

| Model | Default Resolution | Default Frames | Default Steps | Guidance | VRAM (BF16) |
|---|---|---|---|---|---|
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 720x1280 | 81 | 40 | 4.0 | ~60 GiB |
| `robbyant/lingbot-video-dense-1.3b` / `robbyant/lingbot-video-moe-30b-a3b` | 192x320 | 9 | 2 | 3.0 | ~68 GiB (MoE smoke) |
| `Lightricks/LTX-2` | 512x768 | 121 | 40 | video 3.0 / audio 7.0 | Model-dependent |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v` | 480x832 | 121 | 50 | 6.0 | 1×A100 80GB |
| `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` | 720x1280 | 121 | 50 | 6.0 | FP8 + VAE tiling required |
| `Efficient-Large-Model/SANA-Video_2B_480p_diffusers` | 480x832 | 81 | 50 | 6.0 | BF16 DiT + FP32 Wan VAE |
| `Efficient-Large-Model/SANA-Video_2B_720p_diffusers` | 704x1280 | 81 | 50 | 6.0 | ~23.6 GiB |

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

### LingBot-Video

The shared runner recognizes both official LingBot checkpoint IDs and selects
the native `LingBotVideoPipeline` automatically:

```bash
python text_to_video.py \
  --model robbyant/lingbot-video-dense-1.3b \
  --prompt "a robotic arm picks up a red block" \
  --height 192 --width 320 --num-frames 9 --num-inference-steps 2 \
  --guidance-scale 3.0 --flow-shift 3.0 --fps 24 \
  --output lingbot_t2v.mp4
```

Use `robbyant/lingbot-video-moe-30b-a3b` for the MoE checkpoint. For local
paths without `lingbot` in the directory name, add
`--model-class-name LingBotVideoPipeline`. Model-only options use
`--extra-body`, for example `'{"batch_cfg": true, "output_type": "np"}'`.

### LTX-2

```bash
python text_to_video.py \
  --model Lightricks/LTX-2 \
  --prompt "Cherry blossoms swaying gently in the breeze with synchronized ambient sound" \
  --output ltx2_output.mp4
```

See the [LTX-2 recipe](../../../../recipes/LTX/LTX-2.md) for all checkpoints,
pipeline selection, I2V, defaults, and advanced options.

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

### SANA-Video-2B

```bash
python text_to_video.py \
  --model Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  --model-class-name SanaVideoPipeline \
  --prompt "A cinematic tracking shot of a sailboat crossing the ocean at sunset." \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --extra-body '{"motion_score": 30}' \
  --fps 16 \
  --output sana_video_480p.mp4
```

For the 720p checkpoint, switch the model to
`Efficient-Large-Model/SANA-Video_2B_720p_diffusers`, use
`--height 704 --width 1280`, and write to a different output path. SANA-Video
accepts `motion_score`, `clean_caption`, and `use_resolution_binning` through
`--extra-body`. For image-to-video, use the shared
`examples/offline_inference/image_to_video/image_to_video.py` example with
`--model-class-name SanaImageToVideoPipeline` and `--image <path>`.

## Key Arguments

### Common

- `--model`: Diffusers model ID or local path.
- `--model-class-name`: Optional explicit pipeline override.
- `--prompt`: text description (string).
- `--height/--width`: output resolution. Default depends on model.
- `--num-frames`: number of frames. Default depends on model.
- `--guidance-scale`: CFG scale. Default depends on model.
- `--num-inference-steps`: sampling steps. Default depends on model.
- `--fps`: frames per second for the saved MP4.
- `--output`: path to save the generated video.
- `--vae-use-slicing`: enable VAE slicing for memory optimization.
- `--vae-use-tiling`: enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion/parallelism_acceleration.md#cfg-parallel).
- `--tensor-parallel-size`: tensor parallel size (effective for models that support TP, e.g. LTX2).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.
- `--enable-layerwise-offload`: enable layerwise offloading on DiT modules.
- `--frame-rate`: generation FPS for pipelines that require it (e.g., LTX2).
- `--audio-sample-rate`: fallback audio sample rate when the pipeline returns audio.
- `--quantization`: quantization method (`fp8` for FP8, `gguf` for GGUF).
- `--flow-shift`: scheduler flow_shift parameter.
- `--lora-path`: path to PEFT LoRA adapter folder or checkpoint file.
- `--lora-scale`: scale factor for LoRA weights.
- `--lora-backend`: backend for loading LoRA adapters. Default: peft. Available options: peft, distill.

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

## Example materials

??? abstract "text_to_video.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_video/text_to_video.py"
    ``````
