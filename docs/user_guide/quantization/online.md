# Online Quantization

## Overview

Online quantization means vLLM-Omni computes quantized weights and scales while
loading the model. Use it when you want memory savings without preparing a
separate quantized checkpoint.

This mode is different from pre-quantized checkpoint formats such as ModelOpt,
AutoRound, msModelSlim, or serialized Int8 checkpoints. Those formats are
prepared before serving and are documented in their method-specific guides.
For MXFP8 and MXFP4, use this page for load-time quantization from BF16
checkpoints, and use the method-specific pages for offline checkpoints produced
by msModelSlim and the merge tools.

## Hardware Support

| Device | FP8 W8A8 | Int8 W8A8 | BitsAndBytes W4 | MXFP8 W8A8 | MXFP4 W4A4 |
|--------|----------|-----------|------------------|------------|------------|
| NVIDIA Blackwell GPU (SM 100+) | ✅ | ✅ | ✅ | ❌ | ❌ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ | ✅ | ✅ | ❌ | ❌ |
| NVIDIA Ampere GPU (SM 80+) | ✅ | ✅ | ✅ | ❌ | ❌ |
| AMD ROCm (gfx950) | ⭕ | ⭕ | ❌ | ❌ | ✅ |
| Intel XPU | ⭕ | ⭕ | ❌ | ✅ | ❌ |
| Ascend NPU | ❌ | ✅ | ❌ | ✅ | ✅ |

Legend: `✅` backend available, `❌` unsupported, `⭕` not verified in this
guide. These entries describe method-level backend availability, not validation
for every model. FP8 on Ampere may use a weight-only path where available.
On ROCm gfx950, MXFP4 means online single-scale `mxfp4`; online and offline
`mxfp4_dualscale` are Ascend-only. The native MXFP8/MXFP4 configs do not
currently register CUDA implementations; use FP8 or ModelOpt FP8/NVFP4 on
NVIDIA instead.

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

| Method | Guide | Example models | Status |
|--------|-------|----------------|--------|
| FP8 W8A8 | [FP8](fp8.md) | Qwen-Image; Wan2.2 is not validated | Validated for Qwen-Image family and other DiT models |
| Int8 W8A8 | [Int8](int8.md) | Qwen-Image; Wan2.2 is not validated | Validated for Qwen-Image and Z-Image |
| BitsAndBytes W4 | [BitsAndBytes](bitsandbytes.md) | Z-Image-Turbo; Qwen-Image and Wan2.2 are not validated | Validated for Z-Image-Turbo |
| MXFP8 W8A8 | [MXFP8](mxfp8.md) | Wan2.2-T2V-A14B, Wan2.2-I2V-A14B, Wan2.2-TI2V-5B | Validated on Ascend NPU and Intel XPU |
| MXFP4 W4A4 | [MXFP4](mxfp4.md) | Wan2.2-T2V-A14B, Wan2.2-I2V-A14B | Validated on Ascend NPU; the ROCm gfx950 backend is integrated but not model-validated; TI2V-5B is unsupported |

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
omni_bnb = Omni(model="Tongyi-MAI/Z-Image-Turbo", quantization="bitsandbytes")
omni_mxfp8 = Omni(model="<your-model>", quantization="mxfp8")
omni_mxfp4 = Omni(model="<your-model>", quantization="mxfp4")
omni_mxfp4_dualscale = Omni(model="<your-model>", quantization="mxfp4_dualscale")
```

CLI:

```bash
vllm serve <your-model> --omni --quantization fp8
vllm serve <your-model> --omni --quantization int8
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --quantization bitsandbytes
vllm serve <your-model> --omni --quantization mxfp8
vllm serve <your-model> --omni --quantization mxfp4
vllm serve <your-model> --omni --quantization mxfp4_dualscale
```

Per-component routing:

```python
from vllm_omni.quantization import build_quant_config

config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})
```

Component routing is an extension mechanism, not automatic support for every
pipeline component. See [Quantization Scope](overview.md#quantization-scope)
before applying it to an encoder, VAE, or decoder.

## Parameters

| Parameter | Methods | Description |
|-----------|---------|-------------|
| `method` | FP8, Int8, BitsAndBytes, MXFP8, MXFP4 | Quantization method: `"fp8"`, `"int8"`, `"bitsandbytes"`, `"mxfp8"`, `"mxfp4"`, or `"mxfp4_dualscale"` |
| `ignored_layers` | FP8, Int8, BitsAndBytes, MXFP8, MXFP4 | Layer name patterns to keep in BF16/FP16 |
| `activation_scheme` | FP8, Int8 | The runtime value `"dynamic"` selects online activation scaling |
| `weight_block_size` | FP8 | Optional block-wise FP8 weight quantization size |
| `quant_type` | BitsAndBytes | Weight format: `"nf4"` (default) or `"fp4"` |
| `num_bf16_fallback_layers` | MXFP4 DualScale | Leading transformer blocks to keep in BF16 for online `mxfp4_dualscale`; defaults to `5` |

## Validation and Notes

1. Compare the online-quantized output against a BF16 baseline with the same
   seed and generation parameters.
2. Use `ignored_layers` for quality-sensitive MLPs or output projections.
3. Document any required skipped layers in the method page before marking a new
   model as supported.
4. If a model already ships quantized weights, use the matching pre-quantized
   method guide instead of online quantization.
5. For Ascend MXFP4 deployments, prefer offline `mxfp4_dualscale` checkpoints
   when production quality is more important than avoiding preprocessing.
