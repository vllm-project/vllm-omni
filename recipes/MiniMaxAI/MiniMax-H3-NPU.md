# MiniMax H3 on Ascend NPU

> Joint video and audio generation with text, first/last keyframes, and mixed
> image/video/audio references — Ascend NPU deployment guide

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Hardware: Atlas 800I A3
- Maintainer: Community

This recipe adapts [MiniMax-H3.md](MiniMax-H3.md) for Ascend NPU
environments. Differences from the GPU path:

- Runs on a CPU-only (aarch64) PyTorch build with `torch_npu`; no CUDA
  runtime is present.
- Audio loading does **not** require TorchCodec (whose aarch64 wheels are
  built against CUDA torch and fail to load on CPU-only builds). vLLM-Omni
  automatically falls back to soundfile / ffmpeg for wav/mp3/m4a/mp4 audio
  inputs.

## Prerequisites

### Checkpoint

Same as the GPU recipe — Hugging Face access approval is required. Authenticate
once; `vllm serve` downloads the required nested components automatically:

```bash
hf auth login
export MODEL=MiniMaxAI/MiniMax-H3
```

### Environment

- Ascend driver & firmware: 25.5.1
- CANN toolkit: 9.0.1
- Python: 3.12
- PyTorch: 2.10.0+cpu
- torch_npu: 2.10.0.post2
- Install vLLM-Omni from a checkout with MiniMax H3 support:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Install the **mindie-sd** operator library to enable Ascend-optimized fused
operators (`adalayernorm`, etc.) and the RainFusion `rf_v2` block-sparse
attention kernel:

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD

# Comment out the tik_ops build step (not needed for this use case)
sed -i 's|^\(\s*\)source ${current_script_dir}/build_tik_ops.sh|\1# source ${current_script_dir}/build_tik_ops.sh|' build/build_ops.sh

python setup.py bdist_wheel
cd dist
pip install mindiesd-*.whl
```

- `ffmpeg` and `ffprobe` must be available on `PATH`. They are used for
  reference-video preparation and MP4 output.
- Reference-video decoding uses `decord` when available and falls back to
  PyAV otherwise.
- Audio inputs do not require TorchCodec: wav/mp3/m4a/mp4 files are loaded
  at native sample rate through the soundfile / ffmpeg fallback.

## Start a server

Pass the repository ID directly. The pipeline loads the two nested DiTs while
sharing the tokenizer, processor, text encoder, and VAEs from `FL2VA`.

### Multi-NPU: 768P combined-service configuration

Use Ulysses sequence parallelism degree 8, text-encoder tensor parallelism
degree 8, native tiled VAE patch parallelism degree 8, and distributed
layerwise offload:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT=9098
export MODEL=MiniMaxAI/MiniMax-H3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export PYTHONDONTWRITEBYTECODE=1

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --enable-distributed-layerwise-offload \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --diffusion-attention-backend FLASH_ATTN
```

Do not add `--enforce-eager`. The first request includes regional
compilation; warm the server once before measuring steady-state latency.
H3 is CFG-distilled, so `--cfg-parallel-size` must remain 1.

The same endpoint accepts `task=t2va`, `task=fl2va`, and `task=ref2va`; no
partition restart is required. Layerwise offload applies to both DiTs.

On Atlas 800I A3 (64 GB HBM per device) the combined service does not fit at
768P without offloading or sharding: use distributed layerwise offload (as
above) or HSDP — see
[§ Memory and attention optimizations](#memory-and-attention-optimizations-a3).

### CPU MP4 response encoding: MiniMax-H3 FL2VA/t2va (Atlas A2)

This section documents the validated CPU response-encoding behavior for
MiniMax-H3 `FL2VA`/`t2va` on one Atlas A2 host. The implementation itself is
capability-based and has no model whitelist; other models and platforms have
not been tested by this change and have no compatibility or performance claim.

Non-streaming MP4 responses use an automatic encoder. It checks the runtime
frame shape, common dtype, and RGB channel-plane contiguity for every request.
Compatible inputs use direct planar PyAV frames; unsupported inputs use the
legacy fallback, and the worker setting is ignored on that route. Streaming
fMP4 output keeps its existing incremental path. The conservative default is
one CPU frame-conversion worker. The public range is 1 through 8:
`--video-response-frame-conversion-workers 1..8`.

The route log records `selected_path`, `requested_frame_conversion_workers`,
and `effective_frame_conversion_workers`. A legacy fallback records
`effective_frame_conversion_workers=0`. For a direct path with `F` frames and
requested worker count `W`, the effective count is `min(W, F)`. One worker, or
one frame, stays serial and does not create a thread pool.

For more than one worker, each request owns a bounded pool. Conversion futures
are submitted and results are yielded in FIFO order. At most `2 * W_eff`
frames are pending, where `W_eff = min(W, F)`, and each worker reuses its own
thread-local single-channel scratch buffer. Generator close, conversion
failure, and mux failure cancel pending futures and shut down the pool with
`wait=True` and `cancel_futures=True`. The pool is per request, so concurrent
requests can multiply worker, pending-frame, and scratch-buffer resources.

There is no algorithmic worker limit at 8 or 32. The pending-frame bound is
`min(2 * W_eff, F)`. For the validated 1344x768 `gbrp` shape, one PyAV frame
uses 3,096,576 bytes and one float32 single-channel scratch buffer uses
4,128,768 bytes. When the bounded queue is full, the capacity/modelled
maximum of these two known allocation classes is therefore:

`min(2 * W_eff, F) * 3,096,576 + W_eff * 4,128,768` bytes.

This capacity is not guaranteed to be reached at every instant, and it is not
an upper bound for total process memory. It excludes futures, executor
objects, libx264 buffers, the original input, audio, allocator overhead, and
other process state. For the tested worker counts the modelled capacities are:

| Workers | Bounded frame/scratch capacity |
|---:|---:|
| 8 | 78.75 MiB |
| 16 | 157.50 MiB |
| 32 | 315.00 MiB |

With `N` concurrent requests, this per-request capacity model scales
approximately by `N`. Actual process RSS and peak memory are workload
measurements; they cannot be inferred from this model. The public cap is an
operational resource boundary, not a theoretical or universal optimum.

#### Final CPU validation

The final CPU formal used one fixed 124-frame, 1344x768 float32 payload at
24 fps with stereo 32 kHz audio. PyAV/libx264 used `preset=ultrafast` and
`threads=0`. BASE and candidate workers 1, 2, 4, and 8 were each run with one
warmup followed by five formal rounds, in the order `BASE -> 1 -> 2 -> 4 -> 8`.

| Configuration | Wall median (ms) | Process CPU median (ms) | Absolute sampled peak RSS median (KB) |
|---|---:|---:|---:|
| BASE | 6831.562 | 9552.620 | 3168812 |
| Candidate w1 | 6468.999 | 9396.600 | 3167804 |
| Candidate w2 | 4726.955 | 10041.123 | 3188784 |
| Candidate w4 | 3852.694 | 10222.248 | 3223748 |
| Candidate w8 | 3273.143 | 10677.300 | 3296880 |

All 25 formal outputs were byte-identical and passed complete media
validation: H.264 1344x768 at 24 fps with 124 frames, plus AAC stereo audio
at 32 kHz. The CPU results do not measure NPU stage 0 or end-to-end request
latency. Sampled RSS is an absolute process value, not an increment and not an
exact unsampled peak.

Relative to candidate w1, candidate w8 has derived wall change `-49.403%`,
process CPU change `+13.629%`, and RSS change `+4.075%`. The wall percentage is
reported as `(w8 - w1) / w1`, so a negative value means lower wall time.

#### Independent worker-boundary exploration

The same payload was measured independently for workers 8, 16, and 32. Each
worker count had one warmup and five formal rounds in both forward and reverse
order. The forward order was `8 -> 16 -> 32`; the reverse order was
`32 -> 16 -> 8`.

| Workers | Forward wall median (ms) | Reverse wall median (ms) | Forward RSS median (KB) | Reverse RSS median (KB) |
|---:|---:|---:|---:|---:|
| 8 | 2854.683 | 3354.612 | 3293004 | 3297688 |
| 16 | 2855.986 | 3498.202 | 3462104 | 3454060 |
| 32 | 3058.166 | 3482.307 | 3750708 | 3746896 |

All 30 exploratory outputs were byte-identical and passed media validation.
The pooled ten-sample changes were slight and order-sensitive, so they are not
treated as a performance gain. Workers 16 and 32 showed no stable same-order
wall improvement over worker 8 while increasing resource use. This PR keeps
the public range at 1 through 8.

Worker 1 is the conservative default. For the tested single-request
MiniMax-H3/A2 workload, worker 8 is a latency-oriented recommendation when
CPU and memory headroom are available; worker 4 is a lower-resource
intermediate choice. Worker 8 is not a theoretical or general optimum, and
other workloads require fresh measurement.

### Optional optimizations

Two independent optimizations may be enabled on top of the configuration
above. Both are validated for T2VA only.

**RainFusion block-sparse attention** — switch the attention backend, keeping
every other flag unchanged:

```bash
  --diffusion-attention-backend RAINFUSION_ATTN
```

**INT8 online quantization** — add one flag to the server command above:

```bash
  --quantization int8
```

Keep `--ring 1` when using RainFusion: the `rf_v2` kernel ranks key blocks
over the whole sequence, so ring parallelism would split away the keys it
needs. Scale with `--usp` instead.

## Memory and attention optimizations (A3)

### Fitting 768P into 64 GB HBM: distributed layerwise offload or HSDP

Each Atlas 800I A3 NPU has 64 GB of HBM, which is not enough for the
combined MiniMax-H3 service at 768P. Enable **one** of the following at
server startup:

**Distributed layerwise offload** — DiT layers are offloaded with parameters
gathered across the parallel group instead of replicated per rank:

```bash
  --enable-distributed-layerwise-offload
```

Measured peak NPU memory is about 45 GB per device for Ref2VA with a 13.88 s
input video generating a 15 s 768P (1344x768) output video. Host (CPU) memory
usage is high with this option.

**HSDP** — hybrid sharded data parallelism for the DiT parameters. The
multi-stream memory-reuse knob is mandatory with this flag:

```bash
export MULTI_STREAM_MEMORY_REUSE=2
```

```bash
  --use-hsdp
```

Host memory usage is small, but large shapes may still OOM; HBM usage
optimizations for this path are ongoing.

### FLASH_ATTN backend with MindIE-SD

Keep `--diffusion-attention-backend FLASH_ATTN` and install MindIE-SD (see
[§ Environment](#environment)). On NPU this backend carries most of the
memory-reduction and performance work: a mask-free packed varlen path that
never materializes the quadratic `full_qk` padding mask, and K/V prefix
slicing, both driven by the packed `cu_seqlens` metadata emitted by the H3
transformer.

### Optional: LaserAttention fused kernel

For an additional attention speedup, select the Ascend LaserAttention fused
kernel before starting the server:

```bash
export MINDIE_SD_FA_TYPE="ascend_laser_attention"
```

This requires the FLASH_ATTN backend and MindIE-SD. H3 automatically applies
exact power-of-two input pre-scaling (`laser_input_scale=256`) so the
kernel's fp16 workspace cannot overflow on outlier activations. Measured
speedup numbers will be added here.


## HTTP API examples

The request format is identical to the GPU recipe; see
[MiniMax-H3.md § HTTP API examples](MiniMax-H3.md#http-api-examples).
Use the validated 768P shapes (e.g. `width=1344 height=768`) on NPU.

## Key parameters

Same as the GPU recipe; see
[MiniMax-H3.md § Key parameters](MiniMax-H3.md#key-parameters).
The validated resolution on NPU is 768P (e.g. 1344x768).

## Validated NPU evidence

Measured on an Atlas 800I A3 server (8x NPU) with CANN 9.0.1,
PyTorch 2.10.0+cpu, and torch_npu 2.10.0.post2, using the multi-NPU
configuration above:

| Workload | Configuration |
|----------|---------------|
| T2VA, 209 frames, 1344x768 | TE TP8, distributed layerwise offload, Ulysses 8, VPP8 tile, regional compile |
| Ref2VA (prompt + video), 124 frames, 1344x768 | TE TP8, distributed layerwise offload, Ulysses 8, VPP8 tile, regional compile |

These measurements describe the validated shapes rather than a general
throughput guarantee.

## Known limitations

- Combined serving requires sibling `FL2VA` and `Ref2VA` directories, loads
  both task-specific DiTs, and loads one copy of every shared component.
- H3 currently executes one generation request per diffusion batch.
- The first regional-compile request is a warmup and should not be included
  in steady-state performance measurements.
- The official H3 input matrix and media limits are documented in the [GPU
  recipe](MiniMax-H3.md#official-input-matrix-and-limits); this NPU path uses
  the same HTTP request contract.
- VAE patch parallelism requires size 1 or the full DiT group size and
  supports the H3 native `tile` mode only.
- RainFusion block-sparse attention and INT8 quantization are validated for
  T2VA only; use the BF16 dense configuration for FL2VA and Ref2VA.
- Online quantization cannot be combined with distributed layerwise offload
  while AllGather is enabled; pass `--dlo-no-use-allgather` in that case.

## Additional resources

- [MiniMax-H3.md](MiniMax-H3.md) — full GPU guide
- [Attention backends § RAINFUSION_ATTN](../../docs/user_guide/diffusion/attention_backends.md#rainfusion_attn-backend-and-block-sparse-video-attention)
  — RainFusion knobs and tuning
- [Int8 quantization](../../docs/user_guide/quantization/int8.md)
- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
