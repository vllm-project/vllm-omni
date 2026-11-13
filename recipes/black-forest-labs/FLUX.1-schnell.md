# FLUX.1-schnell

> Offline text-to-image generation on one NVIDIA GeForce RTX 5090 32 GB

## Summary

- Vendor: Black Forest Labs
- Model: `black-forest-labs/FLUX.1-schnell`
- Task: Text-to-image generation
- Mode: Offline inference with the shared text-to-image runner
- Maintainer: Community

## When to use this recipe

Use this recipe as a personally validated starting point for generating a
1024x1024 image with FLUX.1-schnell on one NVIDIA GeForce RTX 5090. The tested
configuration uses BF16, four denoising steps and model-level CPU offload.

This recipe covers single-image generation. It does not attempt to optimize
throughput with quantization or cache acceleration.

## References

- Model card: <https://huggingface.co/black-forest-labs/FLUX.1-schnell>
- Canonical offline text-to-image guide:
  [`docs/user_guide/examples/offline_inference/text_to_image.md`](../../docs/user_guide/examples/offline_inference/text_to_image.md)
- Shared runnable example:
  [`examples/offline_inference/text_to_image`](../../examples/offline_inference/text_to_image)
- CPU offload guide:
  [`docs/user_guide/diffusion/cpu_offload.md`](../../docs/user_guide/diffusion/cpu_offload.md)
- Community recipe tracker:
  [#2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

## GPU

### 1x NVIDIA GeForce RTX 5090 32 GB

This configuration was personally validated end to end on September 9, 2026.

#### Environment

- OS: Ubuntu 22.04.3 x86_64
- Python: 3.12.7
- GPU: 1x NVIDIA GeForce RTX 5090, 32,607 MiB reported by `nvidia-smi`
- Host RAM: approximately 62 GiB (64 GB class)
- NVIDIA driver: 595.71.05; CUDA 13.2 reported by `nvidia-smi`
- PyTorch: 2.13.0+cu130; PyTorch CUDA runtime: 13.0
- vLLM: 0.28.0
- vLLM-Omni: `0.28.1.dev112+ga31690041`
- Diffusers: 0.40.0
- Transformers: 5.14.1

#### Command

Run the shared example from the vLLM-Omni repository root in an environment
with vLLM-Omni installed:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model black-forest-labs/FLUX.1-schnell \
  --prompt 'A red fox standing in fresh snow, cinematic photography' \
  --height 1024 --width 1024 \
  --num-inference-steps 4 --guidance-scale 0 --seed 42 \
  --enable-cpu-offload \
  --output flux-schnell-rtx5090.png
```

The command uses the public model ID for portability. The measured runs used
already downloaded FLUX.1-schnell weights; initial Hub download and
authentication were not part of the inference validation.

#### Verification

Verify that the runner succeeded and inspect the saved image:

```bash
python - <<'PY'
import hashlib
from pathlib import Path

from PIL import Image, ImageStat

path = Path("flux-schnell-rtx5090.png")
with Image.open(path) as image:
    image.load()
    assert image.format == "PNG"
    assert image.size == (1024, 1024)
    rgb = image.convert("RGB")
    stats = ImageStat.Stat(rgb)
    assert max(stats.stddev) > 1, "Image is nearly uniform"
    print(f"format={image.format}")
    print(f"size={image.width}x{image.height}")
    print(f"mode={image.mode}")
    print(f"bytes={path.stat().st_size}")
    print("rgb_mean=" + ",".join(f"{value:.3f}" for value in stats.mean))
    print(f"rgb_extrema={rgb.getextrema()}")

print(f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
PY
```

The repeated experiment completed 36 image requests (6 additional warmups and
30 measured outputs) with successful process exits. Every PNG passed decoding,
1024x1024 dimension and non-uniform pixel checks. SHA-256 hashes were recorded
for the artifacts; matching those hashes is not required across environments.
These checks establish file integrity, not a quantitative image-quality score.

#### Performance

The experiment used three independent processes. Each process initialized the
model once, generated two warmup images, then generated ten measured images
sequentially. All 30 measured images used BF16, CPU offload, 1024x1024, four
steps, guidance scale 0 and TP=1, without concurrent requests.

| Generation measurement | Result |
| --- | ---: |
| Measured images | 30 |
| Mean latency | 25.8411 seconds |
| P95 latency (nearest-rank) | 26.1446 seconds |

| Startup / resource measurement | Maximum |
| --- | ---: |
| Model initialization (seconds) | 64.73 |
| Full-lifetime RAM PSS peak (MiB) | 49306 |
| Full-lifetime device GPU memory peak (MiB) | 28121 |
| Measured PyTorch reserved GPU memory peak (MiB) | 27360 |

Latency measures only `Omni.generate()`, excluding image saving and
verification. The two warmup images per process are excluded. P95 uses the
nearest-rank method over the 30 measured images. Model initialization includes
the engine's built-in dummy warmup; weights were already downloaded and OS file
caches were not cleared.

RAM and whole-device GPU memory were sampled throughout each process lifetime
at a target interval of 200 ms. RAM is the maximum simultaneous sum of PSS for
the inference process and its workers, read from Linux `smaps_rollup`.
Whole-device GPU memory is the maximum reported by `nvidia-smi`. Both are
sampled peaks and may miss shorter spikes.

The PyTorch value is the maximum worker-reported `peak_memory_mb` across the
30 measured requests. Despite the field name, its unit is MiB; it measures the
PyTorch allocator's reserved-memory high-water mark rather than whole-device
usage.

#### Notes

- `--enable-cpu-offload` swaps the transformer and text encoders between host
  and device memory at phase boundaries. It reduces GPU residency but adds
  transfer overhead and requires sufficient host RAM. The minimum host RAM
  requirement was not measured.
- CPU offload does not imply CPU-only computation. The tested model-level
  offload backend keeps the VAE on the GPU.
- Only offline single-image generation on one GPU with BF16, 1024x1024,
  four steps and CPU offload was validated. Online serving, FP8, other
  resolutions, concurrency, execution without offload and smaller host RAM
  configurations are outside this recipe's validated scope.
- The upstream revision of the downloaded weights was not recorded, so the
  measurements do not establish exact checkpoint reproducibility.
