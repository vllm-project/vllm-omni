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
  --negative_prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num_frames 48 \
  --guidance_scale 5.0 \
  --guidance_scale_high 6.0 \
  --num_inference_steps 40 \
  --boundary_ratio 0.875 \
  --flow_shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

### Wan2.2-TI2V-5B-Diffusers (Unified)

```bash
python image_to_video.py \
  --model Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --image cherry_blossom.jpg \
  --prompt "Cherry blossoms swaying gently in the breeze, petals falling, smooth motion" \
  --negative_prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num_frames 48 \
  --guidance_scale 4.0 \
  --num_inference_steps 40 \
  --flow_shift 12.0 \
  --fps 16 \
  --output i2v_output.mp4
```

Key arguments:

- `--model`: Model ID (I2V-A14B for MoE, TI2V-5B for unified T2V+I2V).
- `--image`: Path to input image (required).
- `--prompt`: Text description of desired motion/animation.
- `--height/--width`: Output resolution (auto-calculated from image if not set). Dimensions should be multiples of 16.
- `--num_frames`: Number of frames (default 81).
- `--guidance_scale` and `--guidance_scale_high`: CFG scale (applied to low/high-noise stages for MoE).
- `--negative_prompt`: Optional list of artifacts to suppress.
- `--boundary_ratio`: Boundary split ratio for two-stage MoE models.
- `--flow_shift`: Scheduler flow shift (5.0 for 720p, 12.0 for 480p).
- `--num_inference_steps`: Number of denoising steps (default 50).
- `--fps`: Frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--output`: Path to save the generated video.
- `--vae_use_slicing`: Enable VAE slicing for memory optimization.
- `--vae_use_tiling`: Enable VAE tiling for memory optimization.
- `--cfg_parallel_size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](../../../docs/user_guide/diffusion/parallelism_acceleration.md#cfg-parallel).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.

> ℹ️ If you encounter OOM errors, try using `--vae_use_slicing` and `--vae_use_tiling` to reduce memory usage.
