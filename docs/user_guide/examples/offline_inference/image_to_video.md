# Image-To-Video

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_video>.


This example demonstrates how to generate videos from images using Wan2.2,
LTX-2, HunyuanVideo-1.5, and SANA-Video Image-to-Video models with
vLLM-Omni's offline inference API.

## Local CLI Usage

Download the example image:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/cherry_blossom.jpg
```

### Wan2.2-I2V-A14B-Diffusers (MoE)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 5.0 \
  --guidance-scale-high 6.0 \
  --num-inference-steps 40 \
  --boundary-ratio 0.875 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### Wan2.2-TI2V-5B-Diffusers (Unified)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative-prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num-frames 48 \
  --guidance-scale 4.0 \
  --num-inference-steps 40 \
  --flow-shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### LTX-2

```bash
python image_to_video.py \
  --model Lightricks/LTX-2 \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze with synchronized ambient sound" \
  --output ltx2_i2v_output.mp4
```

See the [LTX-2 recipe](../../../../recipes/LTX/LTX-2.md) for all checkpoints,
pipeline selection, T2V, defaults, and advanced options.

### SANA-Video-2B

SANA checkpoints identify their upstream T2V pipeline in `model_index.json`.
Select the native I2V implementation explicitly:

```bash
python image_to_video.py \
  --model Efficient-Large-Model/SANA-Video_2B_480p_diffusers \
  --model-class-name SanaImageToVideoPipeline \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms sway in the breeze as petals drift past the camera." \
  --negative-prompt "blurry, low quality, temporal artifacts" \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --extra-body '{"motion_score": 30}' \
  --fps 16 \
  --seed 42 \
  --output sana_video_i2v_480p.mp4
```

For the 720p checkpoint, use
`Efficient-Large-Model/SANA-Video_2B_720p_diffusers` with
`--height 704 --width 1280`. The native I2V path supports both checkpoint
variants: the 480p checkpoint uses the Wan VAE, while the 720p checkpoint uses
the LTX-2 Video VAE.

For SANA-Video, 81 frames at 16 FPS is the standard checkpoint request
(approximately five seconds), not minute-scale long-video generation.
Minute-scale SANA generation requires the separate LongSANA/LongLive
autoregressive workflow, which this pipeline does not implement.

See the [SANA-Video recipe](../../../../recipes/NVIDIA/SANA-Video-2B.md) for
online serving, backend boundaries, and the tested hardware profile.

Key arguments:

- `--model`: Model ID or local path (for example Wan I2V/TI2V, LTX-2, or
  SANA-Video).
- `--model-class-name`: Explicit pipeline override. SANA-Video I2V requires
  `SanaImageToVideoPipeline`.
- `--image`: Path to input image (required).
- `--prompt`: Text description of desired motion/animation.
- `--height/--width`: Output resolution (auto-calculated from image if not set).
  Wan dimensions should be multiples of 16; LTX dimensions should be multiples
  of 32.
- `--num-frames`: Number of frames (model-specific default; LTX-style models
  work best with `8k + 1`).
- `--guidance-scale` and `--guidance-scale-high`: CFG scale (applied to low/high-noise stages for MoE).
- `--negative-prompt`: Optional list of artifacts to suppress.
- `--boundary-ratio`: Boundary split ratio for two-stage MoE models.
- `--flow-shift`: Scheduler flow shift (5.0 for 720p, 12.0 for 480p).
- `--sample-solver`: Wan2.2 sampling solver. Use `unipc` for the default multistep solver, or `euler` for Lightning/Distill checkpoints.
- `--num-inference-steps`: Number of denoising steps (default 50).
- `--fps`: Frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--audio-sample-rate`: fallback audio sample rate for embedded audio.
- `--output`: Path to save the generated video.
- `--vae-use-slicing`: Enable VAE slicing for memory optimization.
- `--vae-use-tiling`: Enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion/parallelism/cfg_parallel.md).
- `--tensor-parallel-size`: tensor parallel size (effective for models that support TP, e.g. LTX2).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.
- `--use-hsdp`: Enable Hybrid Sharded Data Parallel to shard model weights across GPUs.
- `--hsdp-shard-size`: Number of GPUs to shard model weights across within each replica group. -1 (default) auto-calculates as world_size / replicate_size.
- `--hsdp-replicate-size`: Number of replica groups for HSDP. Each replica holds a full sharded copy. Default 1 means pure sharding (no replication).



> ℹ️ If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.

For Wan2.2 LightX2V-converted local Diffusers directories and related LoRA
assets, see the [LoRA guide](../../diffusion/lora.md#wan22-lightx2v-offline-assembly).

## Example materials

??? abstract "image_to_video.py"
    ``````py
    --8<-- "examples/offline_inference/image_to_video/image_to_video.py"
    ``````
