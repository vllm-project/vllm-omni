# Nucleus-Image

This example shows how to run `NucleusAI/Nucleus-Image` offline with the
shared `Omni(...)` text-to-image interface.

For serving-oriented guidance and H20 notes, see the recipe:
`recipes/NucleusAI/Nucleus-Image.md`.

## Run Example

### Baseline

```bash
python examples/offline_inference/text_to_image/text_to_image.py --model NucleusAI/Nucleus-Image --prompt "A weathered lighthouse on a rocky coastline at golden hour." --height 1024 --width 1024 --num-inference-steps 50 --guidance-scale 4.0 --output nucleus_baseline.png
```

### Parallelism

```bash
# Tensor parallelism
python examples/offline_inference/text_to_image/text_to_image.py --model NucleusAI/Nucleus-Image --tensor-parallel-size 2 --guidance-scale 4.0 --prompt "A cinematic portrait of a red fox in the snow." --output nucleus_tp.png

# Ulysses sequence parallelism
python examples/offline_inference/text_to_image/text_to_image.py --model NucleusAI/Nucleus-Image --ulysses-degree 2 --guidance-scale 4.0 --prompt "A futuristic city skyline at sunrise, volumetric lighting." --output nucleus_ulysses.png

# Ring sequence parallelism
python examples/offline_inference/text_to_image/text_to_image.py --model NucleusAI/Nucleus-Image --ring-degree 2 --guidance-scale 4.0 --prompt "A futuristic city skyline at sunrise, volumetric lighting." --output nucleus_ring.png
```

### Quantization

```bash
python examples/offline_inference/text_to_image/text_to_image.py --model NucleusAI/Nucleus-Image --quantization fp8 --guidance-scale 4.0 --prompt "A studio product photo of a glass perfume bottle." --output nucleus_fp8.png
```

FP8 quantizes eligible transformer linears only. Routed SwiGLU expert weights
remain in checkpoint precision, as do the text encoder and VAE. See the
[FP8 guide](../../../docs/user_guide/quantization/fp8.md) for the exact scope.
This model does not currently advertise Int8 or GGUF support.

## Key Arguments

| Argument | Description |
| :--- | :--- |
| `--tensor-parallel-size` | DiT tensor parallel size. |
| `--ulysses-degree` | Ulysses sequence parallel degree. |
| `--ring-degree` | Ring sequence parallel degree. |
| `--enable-layerwise-offload` | Enable layerwise CPU offload. |
| `--enable-cpu-offload` | Enable CPU offload. |
| `--quantization` | Optional quantization mode, including `fp8`. |
| `--num-images-per-prompt` | Number of output images to save. |

## Notes

- This example is intentionally minimal and focuses on offline text-to-image generation.
- For serving commands, hardware notes, and benchmark-oriented guidance, prefer the recipe under `recipes/NucleusAI/Nucleus-Image.md`.
