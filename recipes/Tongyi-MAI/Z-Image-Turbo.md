# Z-Image Turbo

> Few-step text-to-image with `Tongyi-MAI/Z-Image-Turbo`

## Summary

- Vendor: Tongyi-MAI
- Model: `Tongyi-MAI/Z-Image-Turbo`
- Task: Text-to-image
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to generate 1024x1024 images with the distilled Z-Image Turbo
checkpoint through the shared offline text-to-image example.

## References

- Model card: <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>
- Shared runnable example:
  [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)
- CPU offload guide:
  [`docs/user_guide/diffusion/cpu_offload.md`](../../docs/user_guide/diffusion/cpu_offload.md)

## Hardware Support

This recipe documents one validated Intel XPU configuration. Extend it with
more hardware sections as community validation lands.

## XPU

### 1x Intel Arc Pro B70 (32 GB)

BF16 weights with model-level CPU offload and VAE tiling/slicing, 25 steps.

#### Environment

- OS: Linux
- Python: 3.10+
- torch: 2.13.0+xpu
- vLLM: 0.29.0 (`98dff2a8`)
- vLLM-Omni: `main` at `4c7a98c2`

#### Command

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "a cup of coffee on the table" \
  --num-inference-steps 25 \
  --enable-cpu-offload \
  --vae-use-tiling \
  --vae-use-slicing \
  --enforce-eager \
  --output z_image_turbo_output.png
```

#### Verification

Confirm `z_image_turbo_output.png` is written as a 1024x1024 PNG matching the
prompt.

#### Notes

- Memory usage: peak 15.6 GiB, about 58 s per image.
- Known limitations: only offline generation was qualified. Online serving is
  out of scope for this profile.
