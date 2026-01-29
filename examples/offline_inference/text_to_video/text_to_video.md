# Text-To-Video

The `Wan-AI/Wan2.2-T2V-A14B-Diffusers` pipeline generates short videos from text prompts.

## Local CLI Usage

```bash
python text_to_video.py \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --negative_prompt "<optional quality filter>" \
  --height 480 \
  --width 832 \
  --num_frames 33 \
  --guidance_scale 4.0 \
  --guidance_scale_high 3.0 \
  --flow_shift 12.0 \
  --num_inference_steps 40 \
  --fps 16 \
  --output t2v_out.mp4
```

Key arguments:

- `--prompt`: text description (string).
- `--height/--width`: output resolution (defaults 480x832, i.e. 480P). Dimensions should align with Wan VAE downsampling (multiples of 8).
- `--num_frames`: Number of frames (Wan default is 81).
- `--guidance_scale` and `--guidance_scale_high`: CFG scale (applied to low/high).
- `--negative_prompt`: optional list of artifacts to suppress (the PR demo used a long Chinese string).
- `--boundary_ratio`: Boundary split ratio for low/high DiT. Default `0.875` uses both transformers for best quality. Set to `1.0` to load only the low-noise transformer (saves noticeable memory with good quality, recommended if memory is limited). Set to `0.0` loads only the high-noise transformer (not recommended, lower quality).
- `--fps`: frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--output`: path to save the generated video.
- `--vae_use_slicing`: enable VAE slicing for memory optimization.
- `--vae_use_tiling`: enable VAE tiling for memory optimization.
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.

> ℹ️ If you encounter OOM errors, try using `--vae_use_slicing` and `--vae_use_tiling` to reduce memory usage.
