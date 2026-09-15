# Stable Diffusion 3.5 medium — Intel Arc BMG (XPU)

> Text-to-image with `stabilityai/stable-diffusion-3.5-medium` on Intel XPU

## Summary

- Vendor: Stability AI
- Model: `stabilityai/stable-diffusion-3.5-medium`
- Task: Text-to-image
- Mode: Offline inference (`text_to_image.py`)
- Hardware: 1× Intel Arc BMG (XPU)
- Maintainer: Community (validated via omni-run-artifacts pipeline)

## References

- Model: <https://huggingface.co/stabilityai/stable-diffusion-3.5-medium>
- Datacenter recipe: [`Stable-Diffusion-3.5.md`](./Stable-Diffusion-3.5.md)
- Example: [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)

## Hardware Support

## GPU

### 1× Intel Arc BMG GPU (XPU)

Uses bf16 weights, GPU-only on a single XPU with VAE tiling (28 steps in the
validated run).

#### Command

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model stabilityai/stable-diffusion-3.5-medium \
  --prompt "a sunset over mountains, photorealistic" \
  --num-inference-steps 28 \
  --tensor-parallel-size 1 \
  --vae-use-tiling \
  --output sd35_medium_output.png
```

#### Verification

Confirm `sd35_medium_output.png` exists.
