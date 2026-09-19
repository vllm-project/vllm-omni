# FLUX.2 klein 4B — Intel Arc BMG (XPU)

> Image editing with `black-forest-labs/FLUX.2-klein-4B` on Intel XPU

## Summary

- Vendor: Black Forest Labs
- Model: `black-forest-labs/FLUX.2-klein-4B`
- Task: Image-to-image / editing
- Mode: Offline inference (`image_edit.py`)
- Hardware: 1× Intel Arc BMG (XPU)
- Maintainer: Community (validated via omni-run-artifacts pipeline)

## References

- Model: <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B>
- Example: [`examples/offline_inference/image_to_image/image_edit.py`](../../examples/offline_inference/image_to_image/image_edit.py)

## Hardware Support

## GPU

### 1× Intel Arc BMG GPU (XPU)

Uses bf16 weights with `--enable-cpu-offload` and `--enforce-eager`.

#### Command

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model black-forest-labs/FLUX.2-klein-4B \
  --image test_input.png \
  --prompt "Add a sunset sky with orange and purple clouds" \
  --num-inference-steps 28 \
  --guidance-scale 3.5 \
  --seed 42 \
  --enable-cpu-offload \
  --enforce-eager \
  --output image_edit_output.png
```

Provide any RGB input image path for `--image`.

#### Verification

Confirm `image_edit_output.png` is written successfully.
