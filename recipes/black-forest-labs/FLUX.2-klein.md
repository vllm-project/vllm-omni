# FLUX.2-klein

> Text-guided image editing with `black-forest-labs/FLUX.2-klein-4B`

## Summary

- Vendor: Black Forest Labs
- Model: `black-forest-labs/FLUX.2-klein-4B`
- Task: Image editing (image-to-image)
- Mode: Offline inference
- Maintainer: Community

## When to use this recipe

Use this recipe to run single-image edits with FLUX.2-klein-4B through the
shared offline image-edit example.

## References

- Model card: <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B>
- Shared runnable example:
  [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)
- CPU offload guide:
  [`docs/user_guide/diffusion/cpu_offload.md`](../../docs/user_guide/diffusion/cpu_offload.md)

## Hardware Support

This recipe documents one validated Intel XPU configuration. Extend it with
more hardware sections as community validation lands.

## XPU

### 1x Intel Arc Pro B70 (32 GB)

BF16 weights with model-level CPU offload. `--image` takes any RGB image.

#### Environment

- OS: Linux
- Python: 3.10+
- torch: 2.13.0+xpu
- vLLM: 0.29.0 (`98dff2a8`)
- vLLM-Omni: `main` at `4c7a98c2`

#### Command

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model black-forest-labs/FLUX.2-klein-4B \
  --image test_input.png \
  --prompt "Add a sunset sky with orange and purple clouds" \
  --num-inference-steps 28 \
  --guidance-scale 3.5 \
  --enable-cpu-offload \
  --enforce-eager \
  --seed 42 \
  --output image_edit_output.png
```

#### Verification

Confirm `image_edit_output.png` follows the prompt while preserving the rest
of the input image.

#### Notes

- Memory usage: 7.7 GiB loaded in 8.4 s, about 18 s per edit.
- Known limitations: only offline editing of the 4B checkpoint was qualified.
  `FLUX.2-klein-9B` and online serving are out of scope for this profile.
