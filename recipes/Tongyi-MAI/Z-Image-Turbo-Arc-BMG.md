# Z-Image Turbo — Intel Arc BMG (XPU)

> Text-to-image with `Tongyi-MAI/Z-Image-Turbo` on a single Intel Arc BMG GPU

## Summary

- Vendor: Tongyi-MAI
- Model: `Tongyi-MAI/Z-Image-Turbo`
- Task: Text-to-image
- Mode: Offline inference (`text_to_image.py`)
- Hardware: 1× Intel Arc BMG (XPU), ~32 GiB
- Maintainer: Community (validated via omni-run-artifacts pipeline)

## References

- Model: <https://huggingface.co/Tongyi-MAI/Z-Image-Turbo>
- Example: [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)

## Hardware Support

## GPU

### 1× Intel Arc BMG GPU (XPU)

Qualification uses bf16 weights, module CPU offload, VAE tiling/slicing, and
`--enforce-eager` on Intel XPU (Arc Pro B70 class, ~32 GiB free).

#### Command

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "a cup of coffee on the table" \
  --num-inference-steps 25 \
  --tensor-parallel-size 1 \
  --enable-cpu-offload \
  --vae-use-tiling \
  --vae-use-slicing \
  --enforce-eager \
  --output z_image_turbo_output.png
```

#### Verification

Confirm `z_image_turbo_output.png` exists and looks coherent for the prompt.
