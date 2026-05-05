# Online Quantization

## Overview

Online quantization means vLLM-Omni computes quantized weights and scales while
loading the model. Use it when you want memory savings without preparing a
separate quantized checkpoint.

This mode is different from pre-quantized checkpoint formats such as GGUF,
AutoRound, msModelSlim, or serialized Int8 checkpoints. Those formats are
prepared before serving and are documented in their method-specific guides.

## Hardware Support

| Device | FP8 W8A8 | Int8 W8A8 |
|--------|----------|-----------|
| NVIDIA Blackwell GPU (SM 100+) | ✅ | ✅ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ | ✅ |
| NVIDIA Ampere GPU (SM 80+) | ✅ | ✅ |
| AMD ROCm | ⭕ | ⭕ |
| Intel XPU | ⭕ | ⭕ |
| Ascend NPU | ❌ | ✅ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this
guide. FP8 on Ampere may use a weight-only path where available.

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

| Method | Guide | Example models | Status |
|--------|-------|----------------|--------|
| FP8 W8A8 | [FP8](fp8.md) | Qwen-Image; Wan2.2 is not validated | Validated for Qwen-Image family and other DiT models |
| Int8 W8A8 | [Int8](int8.md) | Qwen-Image; Wan2.2 is not validated | Validated for Qwen-Image and Z-Image |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

Online quantization is not currently validated for the omni/TTS stages. For
Qwen3-Omni and related models, prefer checkpoint-declared ModelOpt or
AutoRound paths when available.

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

Online quantization must be routed to the intended stage. BAGEL and GLM-Image
need model-specific validation before they are listed as supported targets.

## Configuration

Python API:

```python
from vllm_omni import Omni

omni_fp8 = Omni(model="<your-model>", quantization="fp8")
omni_int8 = Omni(model="<your-model>", quantization="int8")
```

CLI:

```bash
vllm serve <your-model> --omni --quantization fp8
vllm serve <your-model> --omni --quantization int8
```

Per-component routing:

```python
from vllm_omni.quantization import build_quant_config

config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})
```

## Parameters

| Parameter | Methods | Description |
|-----------|---------|-------------|
| `method` | FP8, Int8 | Quantization method, usually `"fp8"` or `"int8"` |
| `ignored_layers` | FP8, Int8 | Layer name patterns to keep in BF16/FP16 |
| `activation_scheme` | FP8, Int8 | The runtime value `"dynamic"` selects online activation scaling |
| `weight_block_size` | FP8 | Optional block-wise FP8 weight quantization size |

## Validation and Notes

1. Compare the online-quantized output against a BF16 baseline with the same
   seed and generation parameters.
2. Use `ignored_layers` for quality-sensitive MLPs or output projections.
3. Document any required skipped layers in the method page before marking a new
   model as supported.
4. If a model already ships quantized weights, use the matching pre-quantized
   method guide instead of online quantization.
