# HunyuanVideo 1.5

> Text-to-video with the HunyuanVideo-1.5 Diffusers checkpoints

## Summary

- Vendor: Tencent (community Diffusers ports)
- Model: `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v`
- Task: Text-to-video
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to generate a short clip with HunyuanVideo-1.5 on a memory-
constrained card, trading output size for capacity.

## References

- Model card:
  <https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v>
- Shared runnable example:
  [`examples/offline_inference/text_to_video/text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)
- CPU offload guide:
  [`docs/user_guide/diffusion/cpu_offload.md`](../../docs/user_guide/diffusion/cpu_offload.md)

## Hardware Support

This recipe documents one validated Intel XPU configuration. Extend it with
more hardware sections as community validation lands.

## XPU

### 1x Intel Arc Pro B70 (32 GB)

BF16 weights with model-level CPU offload at a reduced 320x240, 17-frame
shape. The 480p and 720p T2V and I2V checkpoints all use the same command; only
`--model` changes.

#### Environment

- OS: Linux
- Python: 3.10+
- torch: 2.13.0+xpu
- vLLM: 0.29.0 (`98dff2a8`)
- vLLM-Omni: `main` at `4c7a98c2`

#### Command

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --height 240 --width 320 --num-frames 17 \
  --num-inference-steps 50 \
  --guidance-scale 6.0 \
  --enable-cpu-offload \
  --vae-use-tiling \
  --vae-use-slicing \
  --output hunyuan_video_15_output.mp4
```

#### Verification

Confirm `hunyuan_video_15_output.mp4` decodes and that the sampled frames match
the prompt.

#### Notes

- Memory usage: 18.3 GiB loaded, 25.7 GiB peak, about 39 s per clip.
- Known limitations: the native 720p geometry does not fit a 32 GB card with
  CPU offload. Online serving is out of scope for this profile.
