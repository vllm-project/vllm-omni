# Image-To-Video

This example demonstrates how to generate videos from images using Wan2.2 Image-to-Video models with vLLM-Omni's offline inference API.

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

Key arguments:

- `--model`: Model ID (I2V-A14B for MoE, TI2V-5B for unified T2V+I2V).
- `--image`: Path to input image (required).
- `--prompt`: Text description of desired motion/animation.
- `--height/--width`: Output resolution (auto-calculated from image if not set). Dimensions should be multiples of 16.
- `--num-frames`: Number of frames (default 81).
- `--guidance-scale` and `--guidance-scale-high`: CFG scale (applied to low/high-noise stages for MoE).
- `--negative-prompt`: Optional list of artifacts to suppress.
- `--boundary-ratio`: Boundary split ratio for two-stage MoE models.
- `--flow-shift`: Scheduler flow shift (5.0 for 720p, 12.0 for 480p).
- `--num-inference-steps`: Number of denoising steps (default 50).
- `--fps`: Frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--output`: Path to save the generated video.
- `--vae-use-slicing`: Enable VAE slicing for memory optimization.
- `--vae-use-tiling`: Enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](../../../docs/user_guide/diffusion/parallelism_acceleration.md#cfg-parallel).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.

> ℹ️ If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.
