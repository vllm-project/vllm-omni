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

## Automatic Safetensors Startup Staging

Eligible CUDA safetensors loads automatically use bounded pinned-memory staging
(typically 1–3 GiB of pinned host memory per rank). Floating-point weights are
converted to their destination dtype while copied into the staging pool,
avoiding a pageable or dtype-converting H2D transfer. The pinned staging path
is automatically skipped for quantized weights using torchao strategies,
non-floating tensors, CPU destination offload, layerwise-offload backends,
request-scoped pipelines, DeepSpeed CPU offload, and non-quantized HSDP with
CPU destinations. All excluded paths fall back silently to the existing loader
without affecting correctness.

With the multiprocessing executor, the parent process also prewarms checkpoint
pages while GPU workers spawn and import modules. It reads pipeline components
first, then transformer shards. Before worker model initialization starts, all
workers wait for the parent readers to stop and join; foreground component
loading therefore never competes with speculative I/O. The exact model revision
is resolved from the same local Hugging Face cache used by demand loading.
Resident files, insufficient host or cgroup memory, and unavailable local
snapshots degrade to no-op behavior without error.

For tensor-parallel loads, ranks derive the same deterministic bucket schedule.
One rank stages and uploads each bucket before broadcasting it to the other TP
ranks. If every participant receives a reported collective failure, all ranks
replay the unyielded window and remaining stream through local staging.
Checkpoint source failures abort the group instead of being treated as a
successful fallback.

No additional configuration is required. Parseable `[StartupTiming]` logs
report pipeline construction, weight application, worker model load, engine
initialization, warmup, and shutdown. The staging microbenchmark is available
at `benchmarks/diffusion/bench_weight_load_staging.py`.
