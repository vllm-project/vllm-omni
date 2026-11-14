# Diffusion Startup and Loading

Large diffusion models can take several minutes to load at startup. vLLM-Omni
loads safetensors shards in parallel to reduce this initialization time.

Multi-thread weight loading is enabled by default with four threads. No
configuration is needed for the default behavior.

## Configuration

| Parameter | CLI flag | Default | Description |
| --- | --- | --- | --- |
| `enable_multithread_weight_load` | `--disable-multithread-weight-load` | `True` | Pass the flag to disable multi-thread loading. Tensor-parallel cooperative staging bypasses it when the checkpoint is not fully resident |
| `num_weight_load_threads` | `--num-weight-load-threads` | `4` | Number of parallel weight-loading threads |
| `enable_weights_prewarm` | `--enable-weights-prewarm` | `False` | Opt-in page-cache prewarm; measure first, it slows hosts whose construction reads the same disk |

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

Eligible CUDA safetensors loads automatically use bounded pinned-memory staging:
the cooperative path keeps up to 4 GiB of that buffer per rank in flight, while
the single-rank pinned path uses a smaller pool. Floating-point weights are
converted to their destination dtype while copied into the staging pool,
avoiding a pageable or dtype-converting H2D transfer. Non-floating tensors are
still staged, without a dtype cast. The pinned staging path is automatically
skipped for quantized weights using torchao strategies, CPU destination offload,
layerwise-offload backends, and non-quantized HSDP with CPU destinations. For
tensor-parallel loads it is also skipped when the whole checkpoint is already in
the page cache: nothing is left for staging to win once the pages are warm, so
the existing multi-thread loader runs instead. A single-GPU load still uses
pinned staging when the pages are warm. That residency decision is a group
vote, and it travels in the same preflight all-reduce that decides whether
every rank can pin, so no rank can act on it alone. Which components participate
is agreed once per load: the workers exchange their checkpoint source lists and
cooperate only for a component that every rank reported *and* that the model did
not mark private, logging any disagreement.
A component that only some ranks load (the MiniMax-H3 text encoder is sharded over
``text_encoder_tp_size`` ranks, not the DiT group) therefore loads per rank and
never joins a staging collective: no peer holds the tensors it would broadcast.
All excluded paths fall back silently
to the existing loader without affecting correctness.

With the multiprocessing executor, the parent process can also prewarm checkpoint
pages while GPU workers spawn and import modules. It is **off by default**; the rest
of this paragraph describes what enabling it does. One rank per node prewarms
(the page cache is node-shared, so a second reader on the same node only breaks
the first one's readahead); a tensor-parallel group that spans nodes therefore
gets one prewarming rank on each node. Daemon threads issue chunked
``readahead`` / ``posix_fadvise`` hints (falling back to a copy) with the
transformer shards first. Demand loading yields individual safetensors and
parks the readers (no hint and no copy, same offset) while a demanded file
still needs the disk, so foreground component loading is not fighting the
prewarm for I/O. The parent stops the readers once the workers report ready.
Prewarm keeps running during pipeline construction. That is a win when
construction does little checkpoint I/O, and a loss when ``__init__`` itself
reads the text encoder and VAE from a cold disk: on one `g7e.12xlarge`, Wan2.2
TP=2 construction grew from 9 s to 23 s and ready time was slower than a build
without prewarm. The same code was faster on hosts whose construction did not
saturate the disk. Prewarm is therefore **opt-in**: pass ``--enable-weights-prewarm``
(or set ``enable_weights_prewarm=True``) once you have measured that construction
does little checkpoint I/O on your model and storage.
The exact model revision is resolved from the same local Hugging Face cache
used by demand loading.
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
