# Text-To-Video

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_video>.


The `Wan-AI/Wan2.2-T2V-A14B-Diffusers` pipeline generates short videos from text prompts.

## Local CLI Usage

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

Key arguments:

- `--prompt`: text description (string).
- `--height/--width`: output resolution (defaults 480x832, i.e. 480P). Dimensions should align with Wan VAE downsampling (multiples of 8).
- `--num-frames`: Number of frames (Wan default is 81).
- `--guidance-scale` and `--guidance-scale-high`: CFG scale (applied to low/high).
- `--negative-prompt`: optional list of artifacts to suppress (the PR demo used a long Chinese string).
- `--boundary-ratio`: Boundary split ratio for low/high DiT. Default `0.875` uses both transformers for best quality. Set to `1.0` to load only the low-noise transformer (saves noticeable memory with good quality, recommended if memory is limited). Set to `0.0` loads only the high-noise transformer (not recommended, lower quality).
- `--fps`: frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--output`: path to save the generated video.
- `--vae-use-slicing`: enable VAE slicing for memory optimization.
- `--vae-use-tiling`: enable VAE tiling for memory optimization.
- `--cfg-parallel-size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](https://github.com/vllm-project/vllm-omni/tree/main/docs/user_guide/diffusion/parallelism_acceleration.md#cfg-parallel).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.

> ℹ️ If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.

## Example materials

??? abstract "text_to_video.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_video/text_to_video.py"
    ``````
