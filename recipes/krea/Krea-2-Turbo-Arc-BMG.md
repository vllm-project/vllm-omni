# Krea 2 Turbo — Intel Arc BMG (XPU)

> Text-to-image with `krea/Krea-2-Turbo` on Intel XPU

## Summary

- Vendor: Krea
- Model: `krea/Krea-2-Turbo`
- Task: Text-to-image (distilled, few-step)
- Mode: Offline inference (`text_to_image.py`)
- Hardware: 1× Intel Arc BMG (XPU)
- Maintainer: Community (validated via omni-run-artifacts pipeline)

## References

- Model: <https://huggingface.co/krea/Krea-2-Turbo>
- H100 recipe: [`Krea-2.md`](./Krea-2.md)
- Example: [`examples/offline_inference/text_to_image/text_to_image.py`](../../examples/offline_inference/text_to_image/text_to_image.py)

## Hardware Support

## GPU

### 1× Intel Arc BMG GPU (XPU)

Uses bf16 weights with CPU or layerwise offload (see published
`omni-run-artifacts/scripts/krea__Krea-2-Turbo/` for the exact flags from the
passing pipeline run). Typical Turbo settings:

#### Command

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model krea/Krea-2-Turbo \
  --prompt "a fox in the snow" \
  --num-inference-steps 8 \
  --tensor-parallel-size 1 \
  --enable-layerwise-offload \
  --enforce-eager \
  --output krea2_turbo_output.png
```

Adjust offload flags to match your card memory; the pipeline-validated script
is the source of truth when upstream flags differ.

#### Verification

Confirm `krea2_turbo_output.png` exists.
