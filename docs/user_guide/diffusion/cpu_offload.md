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
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --enable-cpu-offload
```

### To Support a Model

Implement the `SupportsComponentDiscovery` protocol to declare which
submodules serve as pipeline components (used by offloading, HSDP
sharding, and other framework features):

```python
from typing import ClassVar
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery

class MyPipeline(nn.Module, SupportsComponentDiscovery):
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


### Component offloading for split models (e.g. Cosmos3)

Some models split their transformer into mutually-exclusive *components* that run
in different phases of a single forward pass rather than as separate pipeline
components -- e.g. Cosmos3's understanding (reasoner) component runs once per
generation while the generation (generator) component runs every denoising step.
Such models have no separate text encoder to swap against, so the transformer
owns a small model-local offload path and wraps each phase with
`with self._offload_context(name):`

```python
class Cosmos3VFMTransformer(nn.Module):
    def forward(self, ...):
        with self._offload_context("reasoner"):
            ...  # understanding pass, runs once
        with self._offload_context("generator"):
            ...  # denoising pass, runs every step
```

Model-level offloading then keeps exactly one component GPU-resident at a time
(the other on CPU), reusing the same `SequentialOffloadHook` `.to()` movers. The
pipeline opts in by exposing `enable_omni_model_cpu_offload` (which drives the
transformer's `enable_model_cpu_offload` and pins the VAE). Layerwise offloading
works for these models too -- each component declares its own block container via
`_layerwise_offload_blocks_attrs`.


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
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --enable-layerwise-offload

# Or image-to-video
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni --enable-layerwise-offload
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
- Cold start latency increases because offloaded components must be moved to CPU
  during setup; layerwise offload may add extra weight consolidation and pinning
  work.
- Performance depends on compute cost and H2D bandwidth as well
- Support single GPU only for now


## Distributed Layerwise Offloading

### How It Works

Distributed layerwise offloading extends single-GPU layerwise offloading to
multi-device deployments. In the default mode, each rank stores only
**1/dp_size** of the model weights in host memory; full layer weights are
reconstructed at runtime via **AllGather** on a dedicated communication stream,
overlapped with computation via a fixed double-buffer scheme. The
`--dlo-no-use-allgather` mode instead streams standard-loader rank-local
weights independently and does not shard weights across ranks.

For the implementation design and compatibility matrix, see
[Distributed Layerwise Offload](../../design/feature/distributed_layerwise_offload.md).

**Key features:**
- **Weight sharding + AllGather**: each rank stores 1/dp_size of weights,
  reconstructed per-layer via `all_gather_into_tensor`
- **Fixed double-buffer**: exactly 2 transformer blocks on each device at any
  time, regardless of model size
- **DP multi-concurrency**: N concurrent requests processed in parallel
  (AllGather only gathers weight shards, which are request-independent)
- **mmap weight loading**: weights loaded as mmap views pointing to shared OS
  page cache, eliminating O(dp_size × model_size) RSS during model creation
- **Platform-agnostic**: works on NVIDIA GPU (CUDA/NCCL) and Ascend NPU
  (CANN/HCCL) via vLLM-Omni's platform abstraction

**Execution Flow (per device):**

```
Compute Stream:  [Layer N]          [Layer N+1]          [Layer N+2]
H2D Stream:      [H2D Shard N+1]   [H2D Shard N+2]
AllGather:       [AG N+1]          [AG N+2]
Slot Usage:      Slot0: Layer N    Slot1: Layer N+1
```

### Usage

**CLI:**
```bash
# 4× GPU/NPU with AllGather (recommended)
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4

# Without DLO AllGather (each rank streams standard-loader rank-local weights)
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --data-parallel-size 4 \
  --dlo-no-use-allgather

# With SP instead of DP (long sequences)
vllm serve /path/to/model --omni \
  --enable-distributed-layerwise-offload \
  --usp 4
```

**Python API:**
```python
from vllm_omni import Omni

m = Omni(
    model="/path/to/model",
    enable_distributed_layerwise_offload=True,
    dlo_use_allgather=True,  # default
)
```

### CLI Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--enable-distributed-layerwise-offload` | Enable DLO | `false` |
| `--data-parallel-size N` | Number of DP ranks; DLO weight-sharding group when AllGather is enabled | `1` |
| `--dlo-use-allgather` | Shard + AllGather (saves CPU, requires concurrent requests) | `true` |
| `--dlo-no-use-allgather` | Use standard-loader rank-local weights; no DLO sharding or AllGather | `false` |

### How model weights are loaded (mmap path)

When DLO + AllGather is active, the offloader:

1. **Saves non-persistent buffers** (e.g. RoPE `inv_freq`, timestep `freqs`)
   from the normally-created transformer
2. **Converts to meta device** via `to_empty(device="meta")`, releasing random
   initialization weights
3. **Loads checkpoint weights as mmap views** via `safe_open().get_tensor()`,
   which return views into the OS page cache (shared across all ranks, 0 RSS)
4. **Calls `post_load_weights()`** to apply model-specific dtype conversions
   (e.g. Cosmos3's `time_embedder` → FP32)
5. **Restores non-persistent buffers** from saved copies

This approach requires **zero model-specific code changes** — no pipeline or
transformer modifications are needed.

### OffloadPlan (declarative topology metadata)

Models can optionally declare an `OffloadPlan` class variable to provide
topology metadata (block attribute names, submodules to offload) without
any offload-specific logic:

```python
from vllm_omni.diffusion.offloader import OffloadPlan

class MyPipeline(nn.Module):
    _dit_modules = ["transformer"]
    _offload_plan = OffloadPlan(
        block_attrs={"transformer": ("blocks",)},
        offload_submodules={"context_encoder": "layers"},
    )
```

When not declared, the offloader falls back to `_layerwise_offload_blocks_attrs`
and heuristic attribute search.  This is backward-compatible — existing models
work without any changes.

### DP Multi-concurrency

When `--data-parallel-size > 1` and AllGather is enabled, the scheduler batches
up to `dp_size` requests per denoise step.  Each DP rank processes a different
request while AllGather synchronizes only weight shards (request-independent).

**Requirements for concurrent requests:**
- `num_inference_steps` must be specified explicitly (None is not allowed)
- All concurrent requests must have the same `num_inference_steps` value
- These constraints exist because AllGather is a collective that requires
  every rank to participate at each denoise step

### Limitations
- Online quantization (FP8) is incompatible with mmap loading — falls back to
  regular loading only when the model is outside the DLO+AllGather mmap path;
  online quantization with DLO+AllGather is rejected. Use
  `--dlo-no-use-allgather` or disable online quantization.
- Tensor Parallel > 1 is rejected in the DLO+AllGather mmap path because TP-aware
  weight-loader callbacks are bypassed. The no-AllGather path retains the
  standard TP loader and is experimental.
- HSDP + AllGather is rejected because it would double-shard parameters. HSDP
  with no-AllGather is accepted at configuration level, but has limited
  end-to-end validation.
- `num_inference_steps=None` is not allowed in DP multi-concurrency mode

**Module Discovery**

The offloader discovers pipeline components in two ways:

1. **Protocol-based** (preferred): If the pipeline implements
    `SupportsComponentDiscovery`, its `_dit_modules`, `_encoder_modules`,
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
├── ModelLevelOffloadBackend → uses SequentialOffloadHook (.to() swap)
│                              (delegates to a pipeline's enable_omni_model_cpu_offload
│                               for split models like Cosmos3)
├── LayerWiseOffloadBackend → uses LayerwiseOffloadHook
│                          (single-GPU, full weights on host)
└── DistributedLayerwiseOffloadBackend → uses DistributedLayerwiseOffloadHook
                                         (multi-GPU, sharded AllGather or rank-local mode)
```

Factory function `get_offload_backend()` selects the appropriate backend based on
configuration.

For split models, `ModelLevelOffloadBackend.enable()` detects a pipeline's
`enable_omni_model_cpu_offload` hook and delegates to it; Cosmos3 then swaps its
reasoner/generator components inside the model forward pass.


## Supported Models

| Architecture | Example Models | DiT Class | Model-Level Offload | Layerwise Offload | Distributed Layerwise Offload | Blocks Attrs (Layerwise specific) |
|--------------|----------------|-----------|---------------------|-------------------|-------------------------------|-----------------------------------|
| Flux2Pipeline | `black-forest-labs/FLUX.2-dev` | `Flux2Transformer2DModel` | ✓ | ✓ | - | `"transformer_blocks"`, `"single_transformer_blocks"` |
| LongCatImagePipeline | `meituan-longcat/LongCat-Image` | `LongCatImageTransformer2DModel` | - | ✓ | - | `"transformer_blocks"`, `"single_transformer_blocks"` |
| NextStep11Pipeline | `stepfun-ai/NextStep-1.1` | `NextStepModel` | - | ✓ | - | `"layers"` |
| OvisImagePipeline | `AIDC-AI/Ovis-Image-7B` | `OvisImageTransformer2DModel` | - | ✓ | - | `"transformer"` |
| QwenImagePipeline | `Qwen/Qwen-Image` | `QwenImageTransformer2DModel` | ✓ | ✓ | - | `"transformer_blocks"` |
| StableDiffusionXLPipeline | `stabilityai/stable-diffusion-xl-base-1.0` | `SDXLUNet2DConditionModel` | ✓ | ✓ | - | `"down_blocks"`, `"up_blocks"` |
| StableDiffusion3Pipeline | `stabilityai/stable-diffusion-3.5-medium` | `SD3Transformer2DModel` | - | ✓ | - | `"transformer_blocks"` |
| Wan22I2VPipeline | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `WanTransformer3DModel` | ✓ | ✓ | - | `"blocks"` |
| Wan22Pipeline | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `WanTransformer3DModel` | ✓ | ✓ | - | `"blocks"` |
| SoulXSingerPipeline / SoulXSingerSVCPipeline | `Soul-AILab/SoulX-Singer` | `DiffLlama` (`cfm_decoder.model.diff_estimator`) | ✓ | ✓ | - | `"layers"` |
| BagelPipeline | `ByteDance-Seed/BAGEL-7B-MoT` | `Qwen2MoTModel` | - | ✓ | - | `"layers"`, `"customized modules"` |
| Cosmos3OmniDiffusersPipeline | `nvidia/Cosmos3-Nano`, `nvidia/Cosmos3-Super` | `Cosmos3VFMTransformer`, `Cosmos3LanguageModel` | ✓ | ✓ | ✓ | `"layers"`, `"gen_layers"` |

**Notes:**
- Model-Level Offloading is expected to be supported by all common diffusion models (DiT and encoders) naturally
- Layerwise Offloading requires DiT class to define `_layerwise_offload_blocks_attrs` pointing to transformer blocks
- Distributed Layerwise Offloading reuses the layerwise block topology, but the
  model and parallelism combination still needs validation. AllGather mode
  additionally requires a compatible checkpoint/mmap layout; no-AllGather mode
  retains the standard loader's rank-local tensors. See the
  [DLO design](../../design/feature/distributed_layerwise_offload.md) and
  [Cosmos3 DistOffload recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/cosmos3/Cosmos3-DistOffload.md)
  for usage examples.
