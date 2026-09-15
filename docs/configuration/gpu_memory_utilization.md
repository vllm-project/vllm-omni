# GPU Memory Calculation and Configuration

This guide explains how to calculate GPU memory requirements and properly configure `gpu_memory_utilization` for vLLM-Omni stages.

## Overview

`gpu_memory_utilization` is a cache-sizing budget, not an instruction to reserve
that fraction of VRAM at startup or a hard runtime memory limit. Its effect
depends on the stage:

- Autoregressive and LLM-generation workers calculate a requested memory
  budget. Only stages that expose KV-cache specifications use the remaining
  budget for automatic cache sizing.
- Diffusion stages use it for KV-cache sizing only with
  `diffusion_kv_mode: paged_scheduler`.
- Diffusion stages using the default `dense_legacy` mode do not use it to size
  or pre-allocate memory.

Paged-scheduler diffusion is a model integration capability, not a reservation
mode that can be enabled for every pipeline. It currently supports HunyuanImage3;
OmniVoice does not support it and cannot use it to reserve or cap VRAM.

For example, `0.8` represents an 80% cache-sizing budget on a stage that
supports automatic sizing. It does not guarantee that `nvidia-smi` will show
80% usage after startup, or prevent visible runtime usage from growing or
falling as allocations and caches change.

## How Memory is Calculated

### Memory Allocation Formula

For stages that support automatic KV-cache sizing, vLLM-Omni calculates the
requested memory as:

```text
requested_memory = total_gpu_memory × gpu_memory_utilization
```

This value is a budget from which model weights, activation peaks, and other
non-cache allocations are subtracted. The remaining memory is available to the
KV cache; the full requested amount is not allocated as an idle reservation.

Paged-scheduler diffusion requires:

```text
free_memory ≥ requested_memory
```

If this condition is not met, paged-scheduler diffusion fails to initialize.
On CUDA, ROCm, and MUSA, autoregressive and LLM-generation workers instead cap
their requested budget at the available free memory and log a warning. Other
platform workers may enforce a strict startup check. Diffusion stages in
`dense_legacy` mode skip this calculation.

Setting `kv_cache_memory_bytes` explicitly overrides automatic KV-cache sizing
from `gpu_memory_utilization`. It does not bypass the startup memory checks
described above; paged-scheduler diffusion still runs its profile warmup.

### Memory Components

The total memory used by a stage includes:

1. **Model Weights**: The size of the model parameters loaded on the GPU
2. **KV Cache**: Memory for storing key-value cache during generation
3. **Activation and Workspace Memory**: Temporary memory for intermediate computations
4. **Allocator and CUDA Graph Pools**: Cached PyTorch allocations and captured graph memory
5. **Non-Torch Memory**: Memory allocated by CUDA libraries and other system components

### Example Calculation

For a GPU with 80GB total memory on a stage that supports automatic sizing:

- `gpu_memory_utilization: 0.8` → 64GB memory budget
- `gpu_memory_utilization: 0.6` → 48GB memory budget
- `gpu_memory_utilization: 0.15` → 12GB memory budget

## Setting Up `gpu_memory_utilization`

### Step 1: Determine GPU Memory

First, check your GPU's total memory:

```bash
# Using nvidia-smi
nvidia-smi --query-gpu=memory.total --format=csv

# Or using Python
python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

### Step 2: Estimate Model Memory Requirements

#### For Autoregressive (AR) Stages

AR stage memory commonly includes:

- Large model weights
- KV cache for attention
- Activation buffers

#### For Diffusion Stages

Diffusion stages have a different runtime memory profile from autoregressive
stages. Model weights, shape-dependent activations and workspaces, and CUDA
graph or allocator caches all contribute to their observed usage.

Diffusion pipelines that keep the default `dense_legacy` mode, including
OmniVoice, should size capacity from measured peak usage after representative
warmup and inference rather than `gpu_memory_utilization`.

### Step 3: Consider Multi-Stage Scenarios

When multiple cache-sized stages share the same GPU, keeping the sum of their
`gpu_memory_utilization` values below 1.0 is a useful starting point. It is not
an isolation guarantee: lazy CUDA graph captures, runtime workspaces, and the
PyTorch caching allocator can increase visible memory after startup, while
`dense_legacy` diffusion stages do not use this value at all. Leave headroom
and use GPU-level isolation such as
[NVIDIA MIG](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) when
workloads must not contend for memory.

#### Example: Two stages on GPU 0

```yaml
stages:
  - stage_id: 0
    devices: "0"
    gpu_memory_utilization: 0.6  # 60% cache-sizing budget on GPU 0

  - stage_id: 1
    devices: "0"
    gpu_memory_utilization: 0.3  # 30% cache-sizing budget on GPU 0
    # Total cache-sizing budget: 90% of GPU 0
```

**Important:** If stages run on different GPUs, each cache-sized stage can use
an independent budget. A value of `1.0` leaves no margin outside the profiled
budget for unprofiled runtime growth or co-resident workloads.

### Step 4: Account for Tensor Parallelism

For models that implement tensor-parallel weight sharding, setting
`tensor_parallel_size > 1` splits supported tensors across multiple GPUs and
usually reduces per-GPU weight memory. Not every stage implementation shards
all weights, and the reduction is not necessarily linear, so measure per-GPU
usage for the selected model.

#### Example: 2-way tensor parallelism

```yaml
stages:
  - stage_id: 0
    devices: "0,1"  # Uses both GPUs
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.6  # 60% cache-sizing budget per GPU
```

## Examples

### Qwen3-Omni-MoE on 2x H100-80GB

```yaml
base_config: /path/to/vllm_omni/deploy/qwen3_omni_moe.yaml

stages:
  - stage_id: 0  # Thinker stage with TP=2
    devices: "0,1"
    tensor_parallel_size: 2
    gpu_memory_utilization: 0.6

  - stage_id: 1  # Talker stage
    devices: "1"
    gpu_memory_utilization: 0.3

  - stage_id: 2  # Code2Wav stage
    devices: "0"
    gpu_memory_utilization: 0.1
```

**Note:** These values are stage inputs, not reservations of 48GB, 24GB, and
8GB. Do not add them to predict physical usage or treat them as hard device
quotas. The Code2Wav stage does not expose a KV cache, so its value does not
create an 8GB cache or reservation.

## Troubleshooting

### Warning or error about insufficient free memory

This means the GPU has less free memory than the configured cache-sizing
budget. CUDA, ROCm, and MUSA autoregressive or LLM-generation workers cap their
budget to the available memory and warn; other platform workers may fail the
startup check. Paged-scheduler diffusion fails initialization.

**Solutions:**

1. Free up memory by closing other processes
2. Reduce `gpu_memory_utilization` for this stage
3. Use a GPU with more memory
4. Move the stage to a different GPU

### Error: OOM during inference

The stage initialized but ran out of memory during processing.

**Solutions:**

1. For AR and LLM-generation stages, reduce `max_num_batched_tokens` or
   `max_num_seqs`
2. For cache-sized stages, lower `gpu_memory_utilization` to reduce the KV cache
3. For dense diffusion, reduce model- or workload-specific memory such as
   batch size, request shapes, or CUDA graph captures
4. Enable quantization or offloading if supported

### Memory Not Fully Utilized

Low startup usage is expected when a stage has no automatically sized KV cache.
For cache-sized stages, you can:

1. Increase `gpu_memory_utilization` to allow a larger KV cache
2. Increase `max_num_batched_tokens` for better batching
3. Check if other stages are limiting throughput

Do not treat low startup usage as capacity that another unisolated workload can
safely consume. Measure representative peak usage, including warmup and varied
request shapes.

## Useful formula for Memory Calculation

### KV Cache Memory

The KV cache size depends on:

- Number of cached tokens across sequences
- Number of layers
- Number of key-value heads per rank and head dimension
- Cache data type
- Tensor-parallel sharding or replication
- Cache block size and allocation rounding

For conventional attention with equal key and value head widths, an approximate
per-rank formula before block rounding is:

```text
kv_cache_memory ≈ cached_tokens × num_layers × 2 × kv_heads_per_rank × head_dim × cache_dtype_size
```

The factor of 2 accounts for keys and values. Other layouts, such as MLA, store
different state, and backends may override head slots or add padding. Use the
reported cache specifications, page sizes, and allocation logs for capacity
planning.

### Model Weight Memory

Approximate raw parameter storage before tensor-parallel sharding and
quantization metadata is:

```text
model_memory ≈ num_parameters × dtype_size
```

For example:

- 7B parameters in FP16: ~14GB
- 7B parameters in FP32: ~28GB
- 7B parameters in INT8: ~7GB

### Activation Memory

Activation memory varies with:

- Batch size
- Sequence length
- Model architecture
- CUDA graph capture shapes and backend workspaces

Measure activation and workspace peaks with representative batch sizes, input
shapes, and output lengths; there is no reliable fixed percentage across AR,
audio, image, and video pipelines.
