# Tensor Parallelism Guide

## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Text Encoder Tensor Parallelism](#text-encoder-tensor-parallelism)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

Tensor Parallelism (TP) shards some model weights across multiple GPUs, usually the Linear layers. This enables running large models that don't fit on a single GPU. It's essential for memory-constrained setups or very large models.

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).

!!! note "TP Limitations for Diffusion Models"
    `--tensor-parallel-size` shards only the DiT (Diffusion Transformer) blocks. The `text_encoder` uses the original Transformers implementation, so by default every TP process keeps a full copy of its weights.

    For Hugging Face text encoders that ship a `base_model_tp_plan`, set `--text-encoder-tp-size` to shard the encoder too; see [Text Encoder Tensor Parallelism](#text-encoder-tensor-parallelism). Decoupling the text encoder from the DiT process is tracked in [Issue #771](https://github.com/vllm-project/vllm-omni/issues/771).

---

## Quick Start

### Basic Usage

Simplest working example:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),  # Enable TP
)

outputs = omni.generate(
    "a cat reading a book",
    OmniDiffusionSamplingParams(num_inference_steps=9),
)
```

---

## Example Script

### Offline Inference

Use Python script under `examples/offline_inference`, and enable TP:

```bash
# Text-to-Image with Qwen-Image
python examples/offline_inference/text_to_image/text_to_image.py \
    --model Qwen/Qwen-Image \
    --tensor-parallel-size 2

# Image Editing with Qwen-Image-Edit
python examples/offline_inference/image_to_image/image_edit.py \
    --model Qwen/Qwen-Image-Edit \
    --image input.png \
    --prompt "Edit description" \
    --tensor-parallel-size 2
```

### Online Serving

You can enable tensor parallelism in online serving via `--tensor-parallel-size`:

```bash
# Text-to-Image with Qwen-Image on 2 GPUs
vllm serve Qwen/Qwen-Image --omni --port 8091 \
    --tensor-parallel-size 2

# Text-to-Image with Z-Image (TP=2 only)
vllm serve Tongyi-MAI/Z-Image-Turbo --omni --port 8091 \
    --tensor-parallel-size 2
```

---

## Text Encoder Tensor Parallelism

`--text-encoder-tp-size N` (`DiffusionParallelConfig.text_encoder_tp_size`) shards the text encoder across groups of `N` DiT ranks. For pipelines that load a Hugging Face encoder (currently the Qwen-Image family: Qwen-Image, Qwen-Image-Edit, Qwen-Image-Edit-Plus, Qwen-Image-Layered), the encoder is sharded with the model's own Transformers `base_model_tp_plan`: attention and MLP projections are split, embeddings and norms stay replicated.

```bash
# Qwen-Image on 4x 32 GB GPUs without CPU offload
vllm serve Qwen/Qwen-Image --omni --tensor-parallel-size 4 --text-encoder-tp-size 4
```

Constraints:

- `N` must divide the number of ranks in one data-parallel replica (world size / `data_parallel_size`) and the encoder's attention and KV head counts (Qwen2.5-VL-7B: 28 heads, 4 KV heads, so `N` ∈ {1, 2, 4}).
- The groups tile each data-parallel replica (`[0..N-1], [N..2N-1], ...`) and never span replicas. Every rank gets the full encoder output, so no broadcast is needed.
- MiniMax-H3 uses its own sharded Qwen3-VL encoder for this option; other pipelines ignore it.

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter              | Type | Default | Description                                                                                                                     |
| ---------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `tensor_parallel_size` | int  | 1       | Number of GPUs to shard model weights across. Must divide number of heads.                                                      |
| `text_encoder_tp_size` | int  | 1       | Number of DiT ranks to shard the text encoder across (see [Text Encoder Tensor Parallelism](#text-encoder-tensor-parallelism)). |

-----------|------|---------|-------------|
| `tensor_parallel_size` | int | 1 | Number of GPUs to shard model weights across. Must divide number of heads. |
| `text_encoder_tp_size` | int | 1 | Number of DiT ranks to shard the text encoder across (see [Text Encoder Tensor Parallelism](#text-encoder-tensor-parallelism)). |

---

## Best Practices

### When to Use

**Good for:**

- Large models that don't fit on a single GPU, especially for models with large DiT blocks (transformer layers)
- Memory-constrained environments

**Not for:**

- When maximum throughput is needed and memory is sufficient
- Models with incompatible dimensions (e.g., Z-Image `num_heads=30`, which now supports `tensor_parallel_size=2`)

## Troubleshooting

### Common Issue 1: Out of Memory (OOM)

**Symptoms**: CUDA OOM errors during model loading or inference, process crashes with memory errors

**Solution**:

```python
# Step 1: Enable TP with smallest degree
parallel_config=DiffusionParallelConfig(tensor_parallel_size=2)

# Step 2: If still OOM, increase TP degree
parallel_config=DiffusionParallelConfig(tensor_parallel_size=4)

```

### Common Issue 2: Divisibility Error

**Symptoms**: Error like "Model dimension X not divisible by tensor_parallel_size Y"

**Solutions**:

1. Check model-specific constraints (e.g., Z-Image only supports TP=2)
2. Use a smaller TP size that divides model dimensions
3. Consult [Supported Models](../../diffusion_features.md#supported-models) for compatible TP sizes

---

## Summary

1. ✅ **Enable TP** - Set `--tensor-parallel-size` to reduce per-GPU memory
2. ✅ **Increase TP size** - Only increase if OOM persists
3. ✅ **Shard the text encoder** - Set `--text-encoder-tp-size` for supported pipelines (Qwen-Image family, MiniMax-H3)
