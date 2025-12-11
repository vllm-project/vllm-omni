# Wan2.2 Text-to-Video Offline Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/wan22>. Added in PR [#202](https://github.com/vllm-project/vllm-omni/pull/202).

The `Wan-AI/Wan2.2-T2V-A14B-Diffusers` pipeline generates short videos from text prompts.

## Local CLI Usage

```bash
python text_to_video.py \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --negative_prompt "<optional quality filter (the PR example uses a long Chinese string)>" \
  --height 720 \
  --width 1280 \
  --num_frames 32 \
  --guidance_scale 4.0 \
  --guidance_scale_high 3.0 \
  --num_inference_steps 40 \
  --fps 16 \
  --output t2v_out.mp4
```

Key arguments:
- `--height/--width`: output resolution (defaults 720x1280). Dimensions should align with Wan VAE downsampling (multiples of 8).
- `--num_frames`: must satisfy `num_frames % 4 == 1` (e.g., 5, 9, 13, 17, 32, 81). Script will round if needed.
- `--guidance_scale` and `--guidance_scale_high`: CFG for low/high-noise branches (high defaults to low when omitted).
- `--negative_prompt`: optional list of artifacts to suppress (the PR demo used a long Chinese string).
- `--boundary_ratio`: split point for the dual DiT stack (default `0.875` from the PR).
- `--flow_shift`: scheduler flow shift (use `5.0` for 720p, `12.0` for 480p as recommended upstream).
- `--fps`: frames per second for the saved MP4 (requires `diffusers` export_to_video).

The script also enables VAE tiling/slicing on NPUs automatically.

## Example materials

??? abstract "text_to_video.py"
    ``````py
    --8<-- "examples/offline_inference/wan22/text_to_video.py"
    ``````
