# CPU Offloading for Diffusion Models

## Overview

vLLM-Omni provides two offloading strategies to reduce GPU memory usage for diffusion models:

1. **Model-level (Sequential) Offloading**: Mutual exclusion between DiT model and encoder - only one is on GPU at a time.
2. **Layerwise (Blockwise) Offloading**: Keeps only one transformer block on GPU at a time with compute-memory overlap.

Both strategies use pinned memory for faster CPU-GPU transfers. The strategies are **mutually exclusive** for now - if both are enabled, layerwise takes priority.


## Model-level (Sequential) Offloading

### How It Works

Model-level offloading implements mutual exclusion between DiT transformer and encoder modules using pre forward hooks:

- **When encoders run**: DiT transformer is offloaded to CPU
- **When DiT runs**: Encoders are offloaded to CPU, if more than one dit models, only one loaded on GPU, others get offloaded to CPU.
- **VAE**: Stays resident on GPU

Before each module's forward pass, the hook automatically moves it to GPU while offloading the other module group to CPU. Transfers use pinned memory for speed.

### Usage

**Python API:**
```python
from vllm_omni import Omni

m = Omni(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", enable_cpu_offload=True)
```

**CLI:**
```bash
vllm-omni serve diffusion Wan-AI/Wan2.2-T2V-A14B-Diffusers --enable-cpu-offload
```

### To Support a Model

Implement the `SupportsModuleOffload` protocol to declare which
submodules participate in offloading:

```python
from typing import ClassVar
from vllm_omni.diffusion.models.interface import SupportsModuleOffload

class MyPipeline(nn.Module, SupportsModuleOffload):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "vision_model"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = []  # optional

    def __init__(self):
        super().__init__()
        self.transformer = ...     # DiT — stays on GPU during denoising
        self.text_encoder = ...    # Encoder — offloaded to CPU during denoising
        self.vision_model = ...    # Encoder — offloaded to CPU during denoising
        self.vae = ...             # VAE — always on GPU
```

- `_dit_modules`: attribute names of denoising submodules (kept on GPU
  during the diffusion loop).
- `_encoder_modules`: attribute names of encoder/vision submodules
  (offloaded to CPU during the diffusion loop).
- `_vae_modules`: attribute names of VAE(s) (always kept on GPU, not
  part of the mutual exclusion hooks).
- `_resident_modules`: attribute names of small submodules that must
  stay on GPU during layerwise offloading (e.g. embedders, connectors).
  Optional — defaults to `[]`.

All attribute names support dotted paths for nested submodules
(e.g. `"pipe.transformer"`, `"bagel.time_embedder"`).

Both DiT and encoder lists are needed because the offload hooks use
mutual exclusion: when one group runs, the other moves to CPU.

### Limitations
- Cold start latency increases
- Adds overhead from CPU-GPU transfers between encoder and denoising phases
- Support single GPU only for now


## Layerwise (Blockwise) Offloading

### How It Works

Layerwise offloading keeps only one transformer block on GPU at a time.

As each block completes, the next block is prefetched to GPU while the current block is freed. The pre and forward hooks utilized by layerwise offloading apply a separate CUDA stream (`copy_stream`) to overlap weight transfer with computation, and retain flattened tensors in pinned CPU memory for block parameters re-materialization. Encoders, VAE, and non-block DiT modules (embeddings, norms) always stay on GPU.

**Execution Flow:**

| Block | Pre-forward Hook | Forward | Post-forward Hook |
|-------|------------------|---------|-------------------|
| block-0 | Prefetch block-1 (async) | Compute block-0 | Free block-0 |
| block-1 | Prefetch block-2 (async) | Compute block-1 | Free block-1 |
| ... | ... | ... | ... |
| block-(n-1) | **Prefetch block-0** (async) | Compute block-(n-1) | Free block-(n-1) |

Each transformer block has a `LayerwiseOffloadHook` that prefetches the next block before forward and frees the current block after forward.

Layerwise offloading is primarily recommended for large **video generation models** where the compute cost per block is high enough to effectively overlap with memory prefetch operations. For example, Wan2.2 T2V and I2V pipelines.

### Usage

**Python API:**
```python
from vllm_omni import Omni

# Text-to-video
m = Omni(model="Wan-AI/Wan2.2-T2V-A14B-Diffusers", enable_layerwise_offload=True)

# Or image-to-video
m = Omni(model="Wan-AI/Wan2.2-I2V-A14B-Diffusers", enable_layerwise_offload=True)
```

**CLI:**
```bash
# Text-to-video
vllm-omni serve diffusion Wan-AI/Wan2.2-T2V-A14B-Diffusers --enable-layerwise-offload

# Or image-to-video
vllm-omni serve diffusion Wan-AI/Wan2.2-I2V-A14B-Diffusers --enable-layerwise-offload
```

### To Support a Model

Models must define the blocks attribute name for layerwise offloading:

```python
class WanTransformer3DModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["blocks"]  # Attribute names containing transformer blocks

    def __init__(self):
        self.blocks = nn.ModuleList([...])  # Transformer blocks
```

For models with multiple block types:

```python
class Flux2Transformer2DModel(nn.Module):
    _layerwise_offload_blocks_attrs = ["transformer_blocks", "single_transformer_blocks"]
```

### Limitations
- Cold start latency increases because of
    1) components are loaded to CPU first at the very first during initialization,
    2) weight consolidation and pinning
- Performance depends on compute cost and H2D bandwidth as well
- Support single GPU only for now


### Implementation Notes

**Module Discovery**

The offloader discovers pipeline components in two ways:

1. **Protocol-based** (preferred): If the pipeline implements
    `SupportsModuleOffload`, its `_dit_modules`, `_encoder_modules`,
    `_vae_modules`, and `_resident_modules` class variables are used
    directly.  All attribute names support dotted paths (e.g.
    `"pipe.transformer"`, `"bagel.time_embedder"`) for nested submodules.

2. **Fallback attribute scan**: Otherwise, the offloader scans for
    well-known attribute names:
    - **DiT modules**: `transformer`, `transformer_2`, `dit`, `sr_dit`, `language_model`, `transformer_blocks`, `model`
    - **Encoders**: `text_encoder`, `text_encoder_2`, `text_encoder_3`, `image_encoder`
    - **VAE**: `vae`, `audio_vae`

**Hook System**

Both strategies use vLLM-Omni's hook registry system (`HookRegistry` and `ModelHook`) to register pre/post forward callbacks on modules, enabling automatic swapping without modifying model code.

**Backend Architecture**

```
OffloadBackend (base class)
├── ModelLevelOffloadBackend → uses SequentialOffloadHook
└── LayerWiseOffloadBackend → uses LayerwiseOffloadHook
```

Factory function `get_offload_backend()` selects the appropriate backend based on configuration.


## Supported Models

| Architecture | Example Models | DiT Class | Model-Level Offload | Layerwise Offload | Blocks Attrs (Layerwise specific) |
|--------------|----------------|-----------|---------------------|-------------------|-----------------------------------|
| LongCatImagePipeline | `meituan-longcat/LongCat-Image` | `LongCatImageTransformer2DModel` | - | ✓ | `"transformer_blocks"`, `"single_transformer_blocks"` |
| NextStep11Pipeline | `stepfun-ai/NextStep-1.1` | `NextStepModel` | - | ✓ | `"layers"` |
| OvisImagePipeline | `AIDC-AI/Ovis-Image-7B` | `OvisImageTransformer2DModel` | - | ✓ | `"transformer"` |
| QwenImagePipeline | `Qwen/Qwen-Image` | `QwenImageTransformer2DModel` | ✓ | ✓ | `"transformer_blocks"` |
| StableDiffusion3Pipeline | `stabilityai/stable-diffusion-3.5-medium` | `SD3Transformer2DModel` | - | ✓ | `"transformer_blocks"` |
| Wan22I2VPipeline | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `WanTransformer3DModel` | ✓ | ✓ | `"blocks"` |
| Wan22Pipeline | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `WanTransformer3DModel` | ✓ | ✓ | `"blocks"` |
| BagelPipeline | `ByteDance-Seed/BAGEL-7B-MoT` | `Qwen2MoTModel` | - | ✓ | `"layers"`, `"customized modules"` |

**Notes:**
- Model-Level Offloading is expected to be supported by all common diffusion models (DiT and encoders) naturally
- Layerwise Offloading requires DiT class to define `_layerwise_offload_blocks_attrs` pointing to transformer blocks
