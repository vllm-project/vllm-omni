# Parallelism Acceleration Guide

This guide covers the parallelism methods in vLLM-Omni for speeding up diffusion model inference and reducing per-device memory requirements.

## Supported Methods

| Method | Description |
|--------|-------------|
| **[Tensor Parallelism](tensor_parallel.md)** | Shards DiT weights across GPUs to reduce per-GPU memory |
| **[Sequence Parallelism](sequence_parallel.md)** | Splits sequence dimension across GPUs (Ulysses-SP, Ring-Attention, or hybrid) for high-resolution images and videos |
| **[CFG-Parallel](cfg_parallel.md)** | Runs CFG positive/negative branches on separate GPUs for ~1.8x speedup on guided generation |
| **[VAE Patch Parallelism](vae_patch_parallel.md)** | Distributes VAE decode spatially across GPUs to reduce peak VAE memory |
| **[HSDP](hsdp.md)** | Shards full model weights via PyTorch FSDP2 to enable large-model inference on memory-constrained GPUs |
| **[Expert Parallelism](expert_parallel.md)** | Shards MoE expert blocks across GPUs for MoE models (e.g. HunyuanImage3.0) |

See [Supported Models](../../diffusion_features.md#supported-models) for per-model compatibility.
