# Feature Compatibility

This guide explains the compatibility matrix of different diffusion features in vLLM-Omni. You can use cache methods together with parallelism methods and other features to achieve optimal speed and efficiency.

## Overview

vLLM-Omni supports combining:

- **Cache methods** (TeaCache, Cache-DiT) with **Parallelism methods** (Ulysses-SP, Ring-Attention, CFG-Parallel, Tensor Parallelism)
- **Multiple parallelism methods** together (e.g., Ulysses-SP + Ring-Attention, CFG-Parallel + Sequence Parallelism)
- **LoRA adapters** with most acceleration features
- **CPU offloading** with other memory optimization features

See the feature compatibility matrix in [Table](diffusion_features.md#feature-compatibility)

## Common Combinations

### 1. Cache + Sequence Parallelism (Recommended)

Best for: **Large images (>1536px) or videos**

Combines cache acceleration with sequence parallelism for maximum speedup on single-device-challenging workloads.

**Using TeaCache + Ulysses-SP:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "A beautiful mountain landscape" \
  --cache-backend tea_cache \
  --ulysses-degree 2
```

**Using Cache-DiT + Ring-Attention:**

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "A futuristic city" \
  --cache-backend cache_dit \
  --ring-degree 2
```

### 2. Cache + CFG-Parallel

Best for: **Image editing with Classifier-Free Guidance**

Accelerates both the diffusion process and CFG computation.

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-Edit \
  --prompt "make it sunset" \
  --negative-prompt "low quality, blurry" \
  --image input.png \
  --cache-backend cache_dit \
  --cfg-parallel-size 2 \
  --cfg-scale 4.0
```

### 3. CFG-Parallel + Sequence Parallelism

Best for: **Large resolution image editing with CFG**

Combines both CFG branch splitting and sequence parallelism for maximum GPU utilization.

**CFG-Parallel + Ulysses-SP:**

```bash
python examples/offline_inference/image_to_image/image_edit.py \
  --model Qwen/Qwen-Image-Edit \
  --prompt "transform into autumn scene" \
  --negative-prompt "low quality" \
  --image input.png \
  --cache-backend cache_dit \
  --cfg-parallel-size 2 \
  --ulysses-degree 2 \
  --cfg-scale 4.0
```

### 4. Hybrid Ulysses + Ring + Vae tiling

Best for: **Very large images or videos on multiple devices**

Combines Ulysses-SP (all-to-all) with Ring-Attention (ring P2P) for scalable parallelism.

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "Epic fantasy landscape" \
  --cache-backend cache_dit \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --num-inference-steps 50 \
  --width 2048 \
  --height 2048 \
  --vae-use-tiling
```

### 5. Cache + Tensor Parallelism

Best for: **Large models that don't fit in single GPU memory**

Reduces per-GPU memory usage while maintaining cache acceleration.

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "A cat reading a book" \
  --cache-backend tea_cache \
  --tensor-parallel-size 2 \
  --num-inference-steps 9 \
```

## Online Serving

### Cache + Sequence Parallelism

```bash
# TeaCache + Ulysses-SP
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh": 0.2}' \
  --usp 2

# Cache-DiT + Ring-Attention
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend cache_dit \
  --cache-config '{"Fn_compute_blocks": 1, "max_warmup_steps": 8}' \
  --ring 2
```

### Cache + CFG-Parallel

```bash
vllm serve Qwen/Qwen-Image-Edit --omni --port 8091 \
  --cache-backend cache_dit \
  --cfg-parallel-size 2
```

### Multiple Parallelism Methods

```bash
# CFG-Parallel + Ulysses-SP (4 GPUs total)
vllm serve Qwen/Qwen-Image-Edit --omni --port 8091 \
  --cache-backend cache_dit \
  --cfg-parallel-size 2 \
  --usp 2

# Hybrid Ulysses + Ring (4 GPUs total)
vllm serve Qwen/Qwen-Image --omni --port 8091 \
  --cache-backend cache_dit \
  --usp 2 \
  --ring 2
```


## Limitations

### Incompatibilities

- **TeaCache + Cache-DiT**: These two cache methods cannot be used together. Only one cache backend can be active at a time. Attempting to enable both will result in an error.

### Partial Support

- **Tensor Parallelism — Text Encoder Not Sharded**: TP currently only shards the DiT blocks. Each TP rank retains a **full copy of the text encoder weights**, leading to significant GPU memory overhead proportional to TP degree. Tracked in [Issue #771](https://github.com/vllm-project/vllm-omni/issues/771).

- **CPU Offloading — Two Modes Are Mutually Exclusive**: Model-level offload (`enable_cpu_offload`) and layerwise offload (`enable_layerwise_offload`) cannot be used simultaneously. If both are set, layerwise takes priority and model-level is silently ignored.

- **CPU Offloading — VAE stays on GPU**: Both offloading strategies keep the VAE on GPU at all times. For high-resolution generation, VAE decode can still cause OOM. Mitigate by combining with `vae_use_tiling=True` or VAE Patch Parallelism.

- **VAE Patch Parallelism — DistributedVaeExecutor Required**: VAE Patch Parallelism is only enabled for models that have `DistributedVaeExecutor`. Unsupported models will silently ignore `vae_patch_parallel_size`, and use sequential vae tiling instead.

### Configuration Constraints

- **GPU Count Must Match Parallel Degrees**: Total GPU count must satisfy:
  ```
  total_gpus = ulysses_degree × ring_degree × cfg_parallel_size × tensor_parallel_size
  ```
  Any mismatch will cause a configuration error at startup.

- **VAE Patch Parallel Size ≤ DiT Process Group Size**: `vae_patch_parallel_size` reuses the DiT process group and cannot exceed it. Larger values are automatically clamped with a warning.

- **Model-Specific TP Constraints**: Some models impose divisibility constraints on TP size. For example, Z-Image Turbo (`num_heads=30`) only supports `tensor_parallel_size=2`. Check [Supported Models](diffusion_features.md#supported-models) for per-model constraints.

## Troubleshooting

### Performance Not Scaling

**Symptoms:** Adding more GPUs doesn't improve speed proportionally

**Solutions:**
1. Check GPU communication bandwidth (use `nvidia-smi topo -m`)
2. Reduce parallelism degree if communication overhead is high
3. For very long sequences, prefer Ring-Attention over Ulysses-SP
4. Ensure batch size is large enough to saturate GPUs

### Out of Memory with Parallelism

**Symptoms:** OOM errors when combining methods

**Solutions:**
1. Enable Tensor Parallelism to shard weights
2. Reduce resolution or batch size
3. Combine with memory efficient methods, such as cpu offloading

### Configuration Errors

**Symptoms:** Errors about invalid parallel configuration

**Solutions:**
1. Verify total GPU count matches: `ulysses × ring × cfg × tp`
2. Check model supports all enabled methods
3. Ensure divisibility constraints (e.g., Z-Image TP=1 or 2 only)

## See Also

- [Diffusion Acceleration Overview](diffusion_features.md) - Main acceleration guide
