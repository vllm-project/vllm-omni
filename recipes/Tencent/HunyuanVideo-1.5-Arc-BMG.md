# HunyuanVideo 1.5 — Intel Arc BMG (XPU)

> Text-to-video with HunyuanVideo-1.5 Diffusers checkpoints on Intel XPU

## Summary

- Vendor: Tencent (community Diffusers ports)
- Model: `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v` (720p T2V)
- Task: Text-to-video
- Mode: Offline inference (`text_to_video.py`)
- Hardware: 1× Intel Arc BMG (XPU)
- Maintainer: Community (validated via omni-run-artifacts pipeline)

## References

- Model: <https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v>
- Example: [`examples/offline_inference/text_to_video/text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)

## Hardware Support

## GPU

### 1× Intel Arc BMG GPU (XPU)

Validated with bf16 weights, CPU offload, reduced geometry (320×240, 17 frames),
VAE tiling/slicing, and 50 steps.

#### Command

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --enable-cpu-offload \
  --vae-use-tiling \
  --vae-use-slicing \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --height 240 \
  --width 320 \
  --num-frames 17 \
  --fps 24 \
  --seed 42 \
  --output hunyuan_video_15_output.mp4
```

#### Verification

Confirm `hunyuan_video_15_output.mp4` is produced.
