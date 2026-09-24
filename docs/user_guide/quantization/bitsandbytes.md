# BitsAndBytes W4 Quantization

## Overview

BitsAndBytes 4-bit quantization supports weight-only NF4/FP4 diffusion transformer
inference on CUDA GPUs. It quantizes BF16/FP16 weights at load time; activations
stay in BF16/FP16. No pre-quantized checkpoint is required.

This is an online (dynamic) path: load the normal HuggingFace checkpoint and
quantize during model loading via the `bitsandbytes` CUDA kernels.

## Hardware Support

| Device | Support |
|--------|---------|
| NVIDIA CUDA GPU (SM 75+) | ✅ |
| NVIDIA Blackwell GPU (SM 100+) | ✅ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ |
| NVIDIA Ampere GPU (SM 80+) | ✅ |
| AMD ROCm | ❌ |
| Intel XPU | ❌ |
| Ascend NPU | ❌ |

Legend: `✅` supported, `❌` unsupported. Non-CUDA platforms raise in
`get_quant_method()`; CUDA requires compute capability 7.5+ (`SM 75+`).

Requires the optional `bitsandbytes` package (`pip install bitsandbytes`).

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

| Model | HF models | CUDA | Mode | Recommendation |
|-------|-----------|:----:|------|----------------|
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | Yes | Online W4 weight-only | All heavy linear layers; sensitive embedders stay BF16 |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Not validated | Online W4 weight-only | Compare vs BF16 before enabling |
| Wan2.2 | Wan2.2 diffusion pipelines | Not validated | Online W4 weight-only | Validate before enabling in docs |

Other diffusion models may work if their transformer uses supported linear
layers, but they are not validated in this guide.

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| Qwen3-Omni | Thinker language-model stage | Not validated | Prefer checkpoint-supported ModelOpt FP8 or AutoRound paths |
| Qwen3-TTS | TTS language-model stage | Not validated | No BitsAndBytes TTS stage support is documented |

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| BAGEL | Stage-specific transformer or DiT module | Not validated | Requires explicit stage routing |
| GLM-Image | Stage-specific transformer or DiT module | Not validated | Requires quality comparison with BF16 |

## Configuration

Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="Tongyi-MAI/Z-Image-Turbo", quantization="bitsandbytes")

omni_with_skips = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    quantization_config={
        "method": "bitsandbytes",
        "ignored_layers": ["to_out"],
    },
)

outputs = omni.generate(
    "A cup of coffee on the table",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

CLI:

```bash
python text_to_image.py --model Tongyi-MAI/Z-Image-Turbo --quantization bitsandbytes
python text_to_image.py --model Tongyi-MAI/Z-Image-Turbo --quantization bitsandbytes --ignored-layers "to_out"
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --quantization bitsandbytes
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | - | Quantization method (`"bitsandbytes"`) |
| `quant_type` | str | `"nf4"` | 4-bit data type: `"nf4"` (recommended) or `"fp4"` |
| `compress_statistics` | bool | `True` | Double-quantize block scaling statistics for better accuracy |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16/FP16 |

## Validation and Notes

On Z-Image-Turbo (single GPU, 1024×1024, 50 steps), BitsAndBytes W4 typically
reduces peak VRAM from roughly 24.5 GiB (BF16) to roughly 17 GiB. Compare output
quality against a BF16 baseline before enabling on new models.

Multi-GPU tensor parallelism (`tensor_parallel_size` > 1) is not validated for
BitsAndBytes in diffusion models; each rank quantizes its own weight shard
independently.

If quality regresses, use `ignored_layers` to keep sensitive projections in BF16
(for example `to_out` or `w2`).

```python
omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    quantization="bitsandbytes",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```

Only add a new model to the supported table after comparing the BitsAndBytes
output against a BF16 baseline and documenting any required `ignored_layers`.
