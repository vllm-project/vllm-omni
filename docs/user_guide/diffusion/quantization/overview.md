# Quantization for Diffusion Transformers

vLLM-Omni supports quantization of DiT linear layers to reduce memory usage and accelerate inference.

## Supported Methods

| Method | Guide |
|--------|-------|
| FP8 | [FP8](fp8.md) |

## Device Compatibility

| GPU Generation | Example GPUs | FP8 Mode |
|---------------|-------------------|----------|
| Ada/Hopper (SM 89+) | RTX 4090, H100, H200 | Full W8A8 with native hardware |

Kernel selection is automatic.
