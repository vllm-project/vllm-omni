# FLUX.2-klein-4B

> Offline text-to-image generation on one NVIDIA L40S 48 GB

## Summary

- Vendor: Black Forest Labs
- Model: `black-forest-labs/FLUX.2-klein-4B`
- Task: Text-to-image generation
- Mode: Offline inference with the shared text-to-image runner
- Maintainer: Community

## When to use this recipe

Use this recipe for a personally validated starting point for generating a
1024x1024 image with FLUX.2-klein-4B on one NVIDIA L40S. The shared example's
unoptimized memory table reports a peak above the card's 48 GB capacity, so
this configuration uses model-level CPU offload and VAE tiling for memory
headroom.

This is a conservative compatibility configuration. It does not attempt to
optimize throughput with quantization or cache acceleration.

## References

- Model card: <https://huggingface.co/black-forest-labs/FLUX.2-klein-4B>
- Canonical offline text-to-image guide:
  [`docs/user_guide/examples/offline_inference/text_to_image.md`](../../docs/user_guide/examples/offline_inference/text_to_image.md)
- Shared runnable example:
  [`examples/offline_inference/text_to_image`](../../examples/offline_inference/text_to_image)
- Community recipe tracker:
  [#2645](https://github.com/vllm-project/vllm-omni/issues/2645)

## Hardware Support

## GPU

### 1x NVIDIA L40S 48 GB

This configuration was personally validated end to end on August 23, 2026.

#### Environment

- Platform: Lightning AI Studio on Linux
- Python: 3.12.14
- GPU: 1x NVIDIA L40S, 46,068 MiB reported by `nvidia-smi`
- Driver / runtime: NVIDIA driver 580.173.02; CUDA 13.0 reported by
  `nvidia-smi`
- PyTorch: 2.13.0
- vLLM: 0.27.1
- vLLM-Omni: `0.27.0rc2.dev78+g2fdbf2234`
- Diffusers: 0.38.0
- Transformers: 5.14.1
- Accelerate: 1.12.0
- Pillow: 12.3.0

The GPU was idle before the run, with no other compute processes reported by
`nvidia-smi`.

#### Command

Run the shared example from the vLLM-Omni repository root:

```bash
CUDA_VISIBLE_DEVICES=0 \
python3 examples/offline_inference/text_to_image/text_to_image.py \
  --model black-forest-labs/FLUX.2-klein-4B \
  --prompt "A red fox wearing round blue glasses holds a white sign that clearly reads 'vLLM-Omni L40S', studio photograph, plain gray background." \
  --seed 0 \
  --tensor-parallel-size 1 \
  --num-images-per-prompt 1 \
  --num-inference-steps 4 \
  --guidance-scale 1.0 \
  --height 1024 \
  --width 1024 \
  --enable-cpu-offload \
  --vae-use-tiling \
  --output flux2-klein-4b-l40s.png
```

The four-step, guidance-scale-1.0 settings follow the distilled checkpoint's
recommended inference settings.

#### Verification

Verify that the runner succeeded and inspect the saved image:

```bash
python3 - <<'PY'
import hashlib
from pathlib import Path

from PIL import Image, ImageStat

path = Path("flux2-klein-4b-l40s.png")
with Image.open(path) as image:
    image.load()
    rgb = image.convert("RGB")
    stats = ImageStat.Stat(rgb)
    print(f"format={image.format}")
    print(f"size={image.width}x{image.height}")
    print(f"mode={image.mode}")
    print(f"bytes={path.stat().st_size}")
    print("rgb_mean=" + ",".join(f"{value:.3f}" for value in stats.mean))
    print(f"rgb_extrema={rgb.getextrema()}")

print(f"sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")
PY
```

The validated run produced these runner and offload log excerpts:

```text
Model-level offloading enabled: transformer <-> text_encoder (mutual exclusion)
Model runner: Initialization complete.
Processed prompts: 100%|██████████| 1/1 [00:13<00:00, 13.32s/it]
Total generation time: 13.3234 seconds (13323.44 ms)
Saved generated image to flux2-klein-4b-l40s.png
```

The output integrity check reported:

```text
format=PNG
size=1024x1024
mode=RGB
bytes=1260085
rgb_mean=169.539,160.801,157.142
rgb_extrema=((0, 255), (0, 255), (0, 255))
sha256=bb88fb56547ad342e4df1e1a07f6bebd299afb5c6f25b55f19f66e8fe38ce4d4
```

Visual inspection confirmed a red fox wearing round blue glasses, holding a
white sign with the requested `vLLM-Omni L40S` text against a plain gray
background. No obvious tile-boundary seams were visible in the output.

#### Performance

| Measurement | Result |
| --- | ---: |
| Generation latency | 13.3234 seconds |
| Denoising latency per step | 3.330 seconds |
| Engine initialization | 55.48 seconds |
| Initial model download | 32.93 seconds |
| Full cold-process wall time | 133.38 seconds |
| Worker-reported peak GPU memory | 12,110 MB |
| Sampled device-wide peak GPU memory | 12,707 MiB (12.409 GiB) |

Generation latency is the shared runner's timed `generate()` call after engine
initialization and its dummy warmup. Full wall time also includes Python
startup, the initial model download, model loading, warmup/compilation,
generation, and shutdown.

Device memory was sampled with `nvidia-smi` every 100 ms for the process
lifetime. Because the GPU was idle before launch, the device-wide maximum is
a useful peak for this run, but it is not a process-isolated allocator metric.

#### Notes

- `--enable-cpu-offload` swaps the transformer and text encoder between host
  and device memory at phase boundaries. This reduces peak VRAM but requires
  sufficient host RAM and adds transfer overhead. Peak host RAM was not
  captured in this validation.
- `--vae-use-tiling` decodes the image in smaller regions to reduce VAE decode
  memory. This run validated tiling, but did not compare memory, latency, or
  image quality against a non-tiled run.
- The first download used unauthenticated Hugging Face Hub access. Set
  `HF_TOKEN` when higher rate limits or more reliable repeated downloads are
  needed.
- The run emitted a warning that no registered `PipelineConfig` was resolved,
  then successfully selected and launched one diffusion stage automatically.
- FLUX.2-klein is distilled; this recipe intentionally uses four denoising
  steps and no classifier-free guidance (`guidance_scale=1.0`).
- The model card notes that generated text may be inaccurate or distorted,
  prompt adherence depends strongly on prompt style, and outputs may not
  always match the prompt.
- Only offline text-to-image generation with TP=1, BF16, CPU offload, and VAE
  tiling was personally validated. Online serving, image editing, FP8,
  cache acceleration, execution without offload, and TP=2 are outside this
  recipe's validated scope.
