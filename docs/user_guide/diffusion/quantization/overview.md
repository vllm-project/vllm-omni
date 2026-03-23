# Quantization for Diffusion Transformers

vLLM-Omni supports quantization of DiT linear layers to reduce memory usage and accelerate inference.

## Supported Methods

| Method | Guide |
|--------|-------|
| FP8 | [FP8](fp8.md) |
| Int8 | [Int8](int8.md) |
| GGUF | [GGUF](gguf.md) |

## Device Compatibility for FP8

| GPU Generation | Example GPUs | FP8 Mode |
|---------------|-------------------|----------|
| Ada/Hopper (SM 89+) | RTX 4090, H100, H200 | Full W8A8 with native hardware |

Kernel selection is automatic.

## Device Compatibility for Int8

| Device Type | Generation | Example | Int8 Mode |
|-------------|---------------|-------------------|----------|
| NVIDIA GPU | Ada/Hopper (SM 89+) | RTX 4090, H100, H200 | Full W8A8 with native hardware |
| Ascend NPU | Atlas A2/Atlas A3 | Atlas 800T A2/Atlas 900 A3 | Full W8A8 with native hardware |
