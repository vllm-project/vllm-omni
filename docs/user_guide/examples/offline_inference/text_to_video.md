# Text-To-Video

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_video>.


The `Wan-AI/Wan2.2-T2V-A14B-Diffusers` pipeline generates short videos from text prompts.

## Local CLI Usage

```bash
python text_to_video.py \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --negative_prompt "<optional quality filter>" \
  --height 480 \
  --width 640 \
  --num_frames 32 \
  --guidance_scale 4.0 \
  --guidance_scale_high 3.0 \
  --num_inference_steps 40 \
  --fps 16 \
  --output t2v_out.mp4
```

Key arguments:

- `--prompt`: text description (string).
- `--height/--width`: output resolution (defaults 720x1280). Dimensions should align with Wan VAE downsampling (multiples of 8).
- `--num_frames`: Number of frames (Wan default is 81).
- `--guidance_scale` and `--guidance_scale_high`: CFG scale (applied to low/high)..
- `--negative_prompt`: optional list of artifacts to suppress (the PR demo used a long Chinese string).
- `--cfg_parallel_size`: the number of devices to run CFG Parallel. CFG Parallel is valid only if classifier-free guidance is enabled and `cfg_parallel_size` is set to 2.
- `--boundary_ratio`: Boundary split ratio for low/high DiT.
- `--fps`: frames per second for the saved MP4 (requires `diffusers` export_to_video).
- `--output`: path to save the generated video.

## Example materials

??? abstract "text_to_video.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_video/text_to_video.py"
    ``````
