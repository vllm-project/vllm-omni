# Wan2.2 T2V 14B — Intel Arc BMG (XPU)

> Text-to-video with `Wan-AI/Wan2.2-T2V-A14B-Diffusers` on Intel XPU

## Summary

- Vendor: Wan-AI
- Model: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
- Task: Text-to-video
- Mode: Offline inference (`text_to_video.py`)
- Hardware: 1× Intel Arc BMG (XPU), ~32 GiB
- Maintainer: Community (validated via omni-run-artifacts pipeline)

## References

- Model: <https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers>
- Example: [`examples/offline_inference/text_to_video/text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)
- Datacenter recipe (Blackwell): [`Wan2.2-T2V.md`](./Wan2.2-T2V.md)

## Hardware Support

## GPU

### 1× Intel Arc BMG GPU (XPU)

Uses bf16 weights, layerwise offload, reduced resolution (320×576, 17 frames),
VAE slicing, and `--enforce-eager`.

#### Command

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --negative-prompt "blurry, noisy, artifacts, distorted, low quality" \
  --num-inference-steps 50 \
  --guidance-scale 5.0 \
  --seed 31337 \
  --height 320 \
  --width 576 \
  --num-frames 17 \
  --tensor-parallel-size 1 \
  --enable-layerwise-offload \
  --enforce-eager \
  --flow-shift 12.0 \
  --boundary-ratio 0.875 \
  --vae-use-slicing \
  --output wan22_t2v_output.mp4
```

#### Verification

Confirm `wan22_t2v_output.mp4` is produced and plays without obvious corruption.
