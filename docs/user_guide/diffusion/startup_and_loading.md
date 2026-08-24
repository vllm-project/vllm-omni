# Diffusion Startup and Loading

Large diffusion models can take several minutes to load at startup. vLLM-Omni
loads safetensors shards in parallel to reduce this initialization time.

Multi-thread weight loading is enabled by default with four threads. No
configuration is needed for the default behavior.

## Pinned Weight Staging

Pipelines whose `load_weights` implementation explicitly declares synchronous
checkpoint-tensor consumption can additionally use a fixed 256 MiB pinned
staging slab. The slab turns pageable host-to-device copies into the faster
pinned transfer path while preserving the checkpoint iterator's order, dtype,
shape, and values. Peak staging memory is capped at 256 MiB per eligible
worker. The inactive host allocator cache is released after loading when the
installed PyTorch version exposes either the current or legacy host-cache
cleanup entry point; otherwise that bounded slab can remain cached for the
worker lifetime. Cache release is process-wide and can also evict inactive
pinned blocks cached by other subsystems in that worker.

Pinned staging is currently available for `StableDiffusion3Pipeline` and
`QwenImagePipeline`, and is opt-in. Other pipelines retain the existing loader
until their `load_weights` contract is audited and opted in. Individual tensors
larger than the slab or smaller than 64 KiB pass through unchanged.

The fast path is disabled for CPU, layerwise, or distributed offload, HSDP,
tensor parallelism, quantized loading, TorchAO safetensors loading,
custom/diffusers pipelines, and platforms without pinned-memory support.
Disabling multi-thread weight loading also restores the ordinary pageable path.
Cold startup can remain storage-bound, so the largest benefit is expected when
checkpoint pages are already resident in the operating-system page cache.

```bash
VLLM_OMNI_ENABLE_PINNED_WEIGHT_STAGING=1 \
  vllm serve stabilityai/stable-diffusion-3.5-medium --omni
```

To measure the pageable and pinned copy paths on the target host:

```bash
python benchmarks/diffusion/bench_pinned_weight_staging.py \
  --size-mib 256 --warmups 2 --iterations 10 --dtype bfloat16
```

## Configuration

| Parameter | CLI flag | Default | Description |
| --- | --- | --- | --- |
| `enable_multithread_weight_load` | `--disable-multithread-weight-load` | `True` | Pass the flag to disable multi-thread loading |
| `num_weight_load_threads` | `--num-weight-load-threads` | `4` | Number of parallel weight-loading threads |

Set `VLLM_OMNI_ENABLE_PINNED_WEIGHT_STAGING=1` to opt into pinned staging.

!!! tip

    The default balances startup speed and disk I/O contention. Fast NVMe
    storage may benefit from more threads, while network storage or hard disks
    may not.

## Online Serving

```bash
# Default: multi-thread loading with four threads
vllm serve Qwen/Qwen-Image --omni --port 8091

# Increase the thread count
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni \
  --num-weight-load-threads 8

# Disable multi-thread loading
vllm serve Qwen/Qwen-Image --omni --disable-multithread-weight-load
```

## Offline Inference

```python
from vllm_omni import Omni

# Default: multi-thread loading with four threads
omni = Omni(model="Qwen/Qwen-Image")

# Increase the thread count
omni = Omni(
    model="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    num_weight_load_threads=8,
)
```

## Reference Benchmarks

The following measurements were collected on NVIDIA H800 hardware. Treat them
as reference results rather than a guarantee for other storage or hardware
configurations.

| Model | Sequential loading | Multi-thread loading | Speedup |
| --- | ---: | ---: | ---: |
| **Qwen/Qwen-Image** (53.7 GiB) | 168 s | 27 s | **6.2x** |
| **Wan-AI/Wan2.2-I2V-A14B-Diffusers** (64.5 GiB) | 283 s | 56 s | **5.1x** |
