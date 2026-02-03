# CPU Offloading for Diffusion Model

## Overview

vLLM-Omni provides two offloading strategies to reduce GPU memory usage for diffusion models, allowing you to run larger models on GPUs with limited VRAM:

1. **Model-level (Component) Offloading**: Swaps entire model components (DiT transformer, VAE, encoders) between GPU and CPU.
2. **Layerwise (Blockwise) Offloading**: Keeps only a single or a few transformer blocks on GPU at a time, with compute - memory copy overlap.

Both approaches use pinned memory for faster CPU-GPU transfers. For now, the two offloading strategies could not be used at the same time.


## Model-level CPU Offloading

### Implementation

CPU offload lets the diffusion worker move large model components between GPU and CPU memory on demand. It keeps the DiT transformer resident on GPU only while it is actively running, and swaps it out when encoders modules need the device. This reduces peak VRAM usage so bigger checkpoints run on smaller GPUs, or multiple requests can share the same GPU.

**Execution Flow**:
1. Text encoders run on GPU while the DiT transformer is offloaded to CPU.
2. Before denoising, weights are prefetched back to GPU, honoring pinned-memory copies for speed.
3. After the diffusion step, the transformer returns to CPU and the process repeats as needed.

Transfers use pinned host buffers, and the worker coordinates swaps via mutex-style hooks so components never compete for memory.

### Configuration
You can enable CPU offload in two ways:

1. **Python API**: set `enable_cpu_offload=True`.

```python
from vllm_omni import Omni

if __name__ == "__main__":

    m = Omni(model="Qwen/Qwen-Image",enable_cpu_offload=True)
```

2. **CLI**: pass `--enable-cpu-offload` to the diffusion service entrypoint.

### Limitations
- Cold start latency increases for over one minute for some models(e.g., Qwen-Image)


## Layerwise (Blockwise) Offloading

### Implementation
Layerwise offload operates at transformer block granularity, keeping a single transformer block, or a specified number of blocks, on GPU while others stay in CPU memory.

Unlike full model-wise CPU offload which swaps entire components like DiT and encoders, layerwise offloading applies a sliding window way of loading and offloading weights between gpu and cpu: while block `i` computes, block `i+1` get prefetched asynchronously via pinned memory. In this way, only partial blocks(s) reside on GPU at any moment during inference, so that greatly decrease the memory occupancy.

**Execution Flow**:

1. During model initialization, all components are loaded to CPU first. Then components other than DiT model(s) in the pipeline, such as VAE and encoders, are moved to GPU. The weights of target transformer blocks are collected as contiguous tensors per layer on CPU with pinned memory; and non-block modules (embeddings, norms, etc) in the DiT model are moved to and stay on GPU.
2. The first block(s) are transferred to GPU during initialization of `LayerwiseOffloader`, before the first denoising step of the very first request.
3. As each block executes, the next block prefetches on a separate CUDA stream for compute - memory copy overlap. After execution, the current block is immediately freed from GPU memory.
4. When the last block completes, the first block prefetches for the next denoising step.


Example of hook executions of a DiT model with n layers, by default keep a single layer on GPU:
| Layer (block) idx | forward pre-hook               | forward          | forward post-hook         |
|-------------------|--------------------------------|------------------|---------------------------|
| layer-0           | prefetch layer 1 (copy stream) | compute layer 0  | free layer-0 gpu weights  |
| layer-1           | prefetch layer 2 (copy stream) | compute layer 1  | free layer-1 gpu weights  |
| layer-2           | prefetch layer 3 (copy stream) | compute layer 2  | free layer-2 gpu weights  |
| ...               | ...                            | ...              | ...                       |
| layer-(n-1)       | **prefetch layer 0 (copy stream)** | compute layer (n-1) | free layer (n-1) gpu weights  |


### Configuration

1. **Python API**: set `enable_layerwise_offload=True` and optionally `layerwise_num_gpu_layers`.

```python
from vllm_omni import Omni

if __name__ == "__main__":
    m = Omni(
        model="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        enable_layerwise_offload=True,
        ...
    )
```

2. **CLI**: pass `--enable-layerwise-offload` and `--layerwise-num-gpu-layers` to the diffusion service entrypoint.

### Supported Models

| Architecture | Models | Example HF Models | DiT Model Cls | Blocks Attr Name |
|--------------|--------|-------------------|----------|----------|
| `QwenImagePipeline` | Qwen-Image-Edit | `Qwen/Qwen-Image` | `QwenImageTransformer2DModel` | "transformer_blocks" |
| `Wan22Pipeline` | Wan2.2 | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `WanTransformer3DModel` | "blocks" |

NOTE: Models must define `_layerwise_offload_blocks_attr` class attribute so that the layerwise offloader finds the target transformer blocks.

### Limitations
- Cold start latency increases because of
    1) components are loaded to CPU first at the very first during initialization,  
    2) weight consolidation and pinning
- Performance depends on CPU <-> GPU interconnection (e.g., PCIe bandwidth).
- Support single GPU only for now
