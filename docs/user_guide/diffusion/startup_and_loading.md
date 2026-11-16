# Diffusion Startup and Loading

Large diffusion models can take several minutes to load at startup. vLLM-Omni
loads safetensors shards in parallel to reduce this initialization time.

Multi-thread weight loading is enabled by default with four threads. No
configuration is needed for the default behavior.

## Configuration

| Parameter | CLI flag | Default | Description |
| --- | --- | --- | --- |
| `enable_multithread_weight_load` | `--disable-multithread-weight-load` | `True` | Pass the flag to disable multi-thread loading |
| `num_weight_load_threads` | `--num-weight-load-threads` | `4` | Number of parallel weight-loading threads |

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
