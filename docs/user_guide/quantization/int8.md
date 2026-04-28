# Int8 Quantization

## Overview

Int8 quantization supports W8A8 diffusion transformer inference on CUDA and
Ascend NPU. It can quantize BF16/FP16 weights at load time, or load serialized
Int8 checkpoints that already contain quantized weights and scales.

Only online activation scaling is currently supported.

## Hardware Support

| Device | Support |
|--------|---------|
| NVIDIA Blackwell GPU (SM 100+) | ✅ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ |
| NVIDIA Ampere GPU (SM 80+) | ✅ |
| AMD ROCm | ⭕ |
| Intel XPU | ⭕ |
| Ascend NPU | ✅ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this
guide.

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

| Model | HF models | CUDA | Ascend NPU | Mode | Recommendation |
|-------|-----------|:----:|:----------:|------|----------------|
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Yes | Yes | Online W8A8 | All layers |
| Wan2.2 | Wan2.2 diffusion pipelines | Not validated | Not validated | Online W8A8 | Validate before enabling in docs |
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | Yes | Yes | Online W8A8 | All layers |

Other diffusion models may work if their transformer uses supported linear
layers, but they are not validated in this guide.

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| Qwen3-Omni | Thinker language-model stage | Not validated | Prefer checkpoint-supported ModelOpt FP8 or AutoRound paths |
| Qwen3-TTS | TTS language-model stage | Not validated | No Int8 TTS stage support is documented |

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

omni = Omni(model="<your-model>", quantization="int8")

omni_with_skips = Omni(
    model="<your-model>",
    quantization_config={
        "method": "int8",
        "ignored_layers": ["<layer-name>"],
    },
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

CLI:

```bash
python text_to_image.py --model <your-model> --quantization int8
python text_to_image.py --model <your-model> --quantization int8 --ignored-layers "img_mlp"
vllm serve <your-model> --omni --quantization int8
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | - | Quantization method (`"int8"`) |
| `activation_scheme` | str | `"dynamic"` | `"dynamic"` selects online activation scaling; static is not supported |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16/FP16 |
| `is_checkpoint_int8_serialized` | bool | `False` | Set by checkpoint config when loading serialized Int8 weights |

## Validation and Notes

Int8 quantization can be combined with cache acceleration:

```python
omni = Omni(
    model="<your-model>",
    quantization="int8",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```

Only add a new model to the supported table after comparing the Int8 output
against a BF16 baseline and documenting any required `ignored_layers`.
