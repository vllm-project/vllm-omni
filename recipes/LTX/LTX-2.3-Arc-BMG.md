# LTX-2.3 Diffusers — Intel Arc BMG (XPU)

> Text-to-video with `diffusers/LTX-2.3-Diffusers` on Intel XPU

## Summary

- Vendor: Lightricks / diffusers
- Model: `diffusers/LTX-2.3-Diffusers`
- Task: Text-to-video (with audio components in pipeline)
- Mode: Offline inference (`text_to_video.py`)
- Hardware: 1× Intel Arc BMG (XPU), ~32 GiB
- Maintainer: Community (validated via omni-run-artifacts pipeline)

## References

- Model: <https://huggingface.co/diffusers/LTX-2.3-Diffusers>
- Family (datacenter): [`LTX-2.md`](./LTX-2.md)
- Example: [`examples/offline_inference/text_to_video/text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py)

## Hardware Support

## GPU

### 1× Intel Arc BMG GPU (XPU)

Uses bf16 weights, layerwise offload, 512×768, 81 frames, VAE tiling, and
`--enforce-eager`. On-demand text-encoder staging avoids single-pass OOM on
~32 GiB cards.

#### Command

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model diffusers/LTX-2.3-Diffusers \
  --prompt "A serene lakeside sunrise with mist over the water." \
  --num-inference-steps 30 \
  --height 512 \
  --width 768 \
  --num-frames 81 \
  --fps 24 \
  --tensor-parallel-size 1 \
  --enable-layerwise-offload \
  --vae-use-tiling \
  --enforce-eager \
  --output ltx23_output.mp4
```

#### Verification

Confirm `ltx23_output.mp4` is written successfully.
