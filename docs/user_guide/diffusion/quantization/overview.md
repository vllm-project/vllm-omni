# Quantization

vLLM-Omni provides a unified quantization framework that supports both diffusion models (DiT, text encoders, VAEs) and multi-stage omni models (e.g., Qwen3-Omni thinker). Quantization reduces memory usage and accelerates inference.

## Supported Methods

### Diffusion Models

| Method | Guide | Description | Tested Models | Min GPU |
|--------|-------|-------------|---------------|---------|
| FP8 | [FP8](fp8.md) | FP8 W8A8, dynamic or static | Z-Image, Qwen-Image, Flux, Bagel | SM 89 (Ada) |
| Int8 | [Int8](int8.md) | Int8 W8A8 | Z-Image, Qwen-Image | SM 89 (Ada) / Ascend NPU |
| GGUF | [GGUF](gguf.md) | GGUF format, dequant+GEMM for N-D tensors | Z-Image, Flux | SM 60 |
| AutoRound | [AutoRound](autoround.md) | W4A16 (pre-quantized) | Flux | SM 80 (Ampere) |

### Multi-stage Omni Models (Pre-quantized Checkpoints)

| Method | Description | Tested Models | Min GPU |
|--------|-------------|---------------|---------|
| ModelOpt FP8 | Pre-quantized FP8 via NVIDIA ModelOpt | Qwen3-Omni (thinker) | SM 89 (Ada/Hopper) |
| ModelOpt NVFP4 | Pre-quantized NVFP4 via NVIDIA ModelOpt | Qwen3-Omni (experimental, quality issues) | SM 100 (Blackwell) |

!!! note
    AWQ, GPTQ, and BitsAndBytes are supported by vLLM's upstream quantization registry but have **not been tested** with vLLM-Omni pipelines. They may work via `build_quant_config()` but are not validated.

### Pre-quantized LLM Checkpoints (Multi-stage Models)

For multi-stage models like Qwen3-Omni, the unified quantization framework auto-detects
pre-quantized checkpoints via `quantization_config` in the HF config. Quantization is
automatically scoped to the thinker's `language_model` — audio encoder, vision encoder,
talker, and code2wav remain in BF16.

| Format | `quant_algo` | Hardware | Example | Status |
|--------|-------------|----------|---------|--------|
| ModelOpt FP8 | `FP8` | Ada/Hopper (SM 89+) | `asdazd/Qwen3-Omni-30B-A3B-Instruct_modelopt_FP8` | Tested — 47% memory reduction, comparable throughput |
| ModelOpt NVFP4 | `NVFP4` | Blackwell (SM 100+) | `shunyang90/Qwen3-Omni-30B-A3B-Instruct-NVFP4` | Experimental — loads and runs but output quality is unacceptable |

**Tested FP8 results (Qwen3-Omni, 1×H200):**

| Config | Model Memory (GiB) | Mem Reduction | Decode (tok/s) |
|--------|-------------------|---------------|----------------|
| BF16 baseline | 59.26 | — | 41.6 |
| FP8 (ModelOpt) | 31.41 | 47% | 39.9 |

FP8 enables the full 3-stage pipeline (thinker + talker + code2wav) on a single 64GB GPU,
which is impossible with BF16 (thinker alone requires 59.26 GiB).

## Quantization Scope

### Diffusion Models

When `--quantization fp8` is enabled for diffusion models:

| Component | What Gets Quantized | Mechanism |
|-----------|-------------------|-----------|
| **DiT (transformer)** | `nn.Linear` layers | vLLM W8A8 FP8 compute (Ada/Hopper) or weight-only (older GPUs) |
| **Text encoder** | `nn.Linear` layers | FP8 weight storage, BF16 compute |
| **VAE** | `nn.Conv2d`, `nn.Conv3d` layers | FP8 weight storage, BF16 compute |

### Multi-stage Omni Models

For pre-quantized omni model checkpoints, quantization is scoped per component:

| Component | Quantized? | Notes |
|-----------|-----------|-------|
| **Thinker (language_model)** | Yes | FP8 or NVFP4 via ModelOpt |
| **Audio encoder** | No | Stays BF16 |
| **Vision encoder** | No | Stays BF16 |
| **Talker** | No | Stays BF16 |
| **Code2Wav** | No | Stays BF16 |

!!! note
    Not all models support all components. See the [FP8 supported models table](fp8.md#supported-models) for per-model details.

## Device Compatibility for FP8

| Device | Example Hardware | FP8 | NVFP4 |
|--------|-----------------|-----|-------|
| Blackwell GPU (SM 100+) | RTX 5090, B100, B200 | Yes | Yes (native FP4 HW) |
| Ada/Hopper GPU (SM 89+) | RTX 4090, H100, H200 | Yes (W8A8 native) | No |
| Turing/Ampere GPU (SM 75-86) | RTX 3090, A100 | Yes (weight-only Marlin) | No |
| Ascend NPU | Atlas 800T A2 (910B) | Not yet supported | No |

Kernel selection is automatic on CUDA GPUs.

## Device Compatibility for Int8

| Device Type | Generation | Example | Int8 Mode |
|-------------|---------------|-------------------|----------|
| NVIDIA GPU | Ada/Hopper (SM 89+) | RTX 4090, H100, H200 | Full W8A8 with native hardware |
| Ascend NPU | Atlas A2/Atlas A3 | Atlas 800T A2/Atlas 900 A3 | Full W8A8 with native hardware |

## Python API

The `build_quant_config()` factory accepts multiple input formats:

```python
from vllm_omni.quantization import build_quant_config

# String
config = build_quant_config("fp8")

# Dict with parameters
config = build_quant_config({"method": "fp8", "activation_scheme": "static"})

# AutoRound / INC (auto-detected from checkpoint, or explicit)
config = build_quant_config("auto-round", bits=4, group_size=128)

# Per-component dict
config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})

# None / "none"
config = build_quant_config(None)  # returns None
```

## Architecture

```
build_quant_config(spec)
    ├── str  → _build_single(method) → vLLM registry / _OVERRIDES
    ├── dict with "method" → _build_single(method, **kwargs)
    ├── per-component dict → ComponentQuantizationConfig
    │       routes get_quant_method() by longest-prefix match
    ├── QuantizationConfig → passthrough
    └── None / "none" → None
```

## Migration Guide

### Before (v0.14.x)

```python
# Old diffusion-specific API (removed)
from vllm_omni.diffusion.quantization import get_diffusion_quant_config
config = get_diffusion_quant_config("fp8", activation_scheme="static")
```

### After (v0.16.0+)

```python
# Unified API — delegates to vLLM's registry
from vllm_omni.quantization import build_quant_config
config = build_quant_config({"method": "fp8", "activation_scheme": "static"})

# Per-component (new)
config = build_quant_config({
    "transformer": {"method": "fp8"},
    "vae": None,
})
```
