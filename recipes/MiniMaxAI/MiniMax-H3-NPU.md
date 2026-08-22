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

The FL2VA endpoint accepts `task=t2va` and `task=fl2va`. Serve `task=ref2va`
from a separate Ref2VA server invocation; one process cannot load both
checkpoint partitions. Layerwise offload applies to both FL2VA DiTs.

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
submitted or pending futures represent frames, where `W_eff = min(W, F)`. A
converted PyAV frame yielded to the muxer can coexist with those pending
futures. Each worker also reuses its own thread-local single-channel scratch
buffer, so the model adds `W_eff` such scratch buffers. Generator close,
conversion failure, and mux failure cancel pending futures and shut down the
pool with `wait=True` and `cancel_futures=True`. The pool is per request, so
concurrent requests can multiply worker, pending-frame, and scratch-buffer
resources.

For `W_eff > 1`, the conservative known frame-storage count is
`min(F, 2 * W_eff + 1)`: the extra one is the yielded frame and is distinct
from the `2 * W_eff` pending-future bound. For the validated 124-frame,
1344x768 `gbrp` shape, one PyAV frame uses 3,096,576 bytes and one float32
single-channel scratch buffer uses 4,128,768 bytes. The known-capacity model
is therefore:

`min(F, 2 * W_eff + 1) * 3,096,576 + W_eff * 4,128,768` bytes.

For `W_eff = 1`, conversion is serial, no pool is created, and one converted
frame plus one thread-local scratch buffer is used at a time. For the pooled
worker counts below, the pending-future column excludes the yielded frame, and
the capacity includes both frame classes:

| Workers | Pending-future bound | Known frame slots | Known capacity |
| ---: | ---: | ---: | ---: |
| 8 | 16 | 17 | 81.703125 MiB |
| 16 | 32 | 33 | 160.453125 MiB |
| 32 | 64 | 65 | 317.953125 MiB |

This is a conservative known-capacity model, not total process memory. It is
not guaranteed to be reached at every instant and excludes futures, executor
objects, libx264 buffers, the original input, audio, allocator overhead, and
other process state. With `N` concurrent requests, this per-request model
scales approximately by `N`; actual process RSS and peak memory require
workload measurement.

The public cap is an operational resource boundary, not a theoretical or
universal optimum.

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

The same payload was measured for workers 8, 16, and 32. Each worker count had
one warmup and five formal rounds in both forward and reverse order. The
forward order was `8 -> 16 -> 32`; the reverse order was `32 -> 16 -> 8`.

Workers 16 and 32 were measured in an independent cap-unlocked experimental
worktree using the same bounded algorithm. The public CLI and programmatic
validator reject values above 8, so the public operational range is 1 through
8. The bounded queue imposes no algorithmic or theoretical limit at 8.

| Workers | Forward wall median (ms) | Reverse wall median (ms) | Forward RSS median (KB) | Reverse RSS median (KB) |
|---:|---:|---:|---:|---:|
| 8 | 2854.683 | 3354.612 | 3293004 | 3297688 |
| 16 | 2855.986 | 3498.202 | 3462104 | 3454060 |
| 32 | 3058.166 | 3482.307 | 3750708 | 3746896 |

All 30 exploratory outputs were byte-identical and passed media validation.
The pooled ten-sample changes were slight and order-sensitive, so they are not
treated as a performance gain. Workers 16 and 32 showed no stable same-order
wall improvement over worker 8 while increasing resource use. The public range
remains 1 through 8.

#### Final A2 E2E validation (measured)

This final measured E2E run was scoped strictly to MiniMax-H3 `FL2VA`/`t2va`:
one request at `1344x768@24fps`, fixed Laser configuration, on one host with
8x Atlas A2 910B4-1. BASE was `bd4f9acfd30456cb8fa98af53d32f7adc34e03a0`
without PR2; the candidate runtime was
`81804c86f51bb8ab31827bbf3dbd2a62ef03bee3`; the run date was
`2026-08-22`. The model partition path is represented as
`$MODEL_ROOT/FL2VA`.

BASE and candidate were run serially. Each side had one warmup and three
formal requests for each 5, 8.7, and 15 second duration. The fixed prompt was:

> In a snowy blue-purple forest, Ori carefully walks past a sleeping giant;
> footsteps crunch in the snow while the creature breathes and softly snorts.

The seed was 1101. Each request requested 50 steps and executed 49 DiT
forwards. BASE and candidate both used the known-good eight-NPU command above
with `MODEL=$MODEL_ROOT/FL2VA`,
`MINDIE_SD_FA_TYPE=ascend_laser_attention`, and
`--diffusion-attention-backend FLASH_ATTN`. Candidate alone additionally used
`--video-response-frame-conversion-workers 8`; BASE did not use the PR2 flag.
No Laser operator trace was captured, so Laser activation is not
trace-confirmed.

The measured environment was:

| Component | Version / provenance |
| --- | --- |
| Driver / firmware | `25.5.2` / `7.8.0.7.220` |
| CANN | `9.0.1 V100R001C10SPC002B220` |
| Python | `3.12.13` |
| PyTorch / `torch_npu` | `2.10.0+cpu` / `2.10.0.post2` |
| vLLM | `0.26.0+empty` |
| vLLM-Ascend | `0.19.1rc2.dev1251+g905bbf372` |
| vLLM-Omni metadata | `0.26.1.dev138+g596c16a55.npu` |
| Import origins | Exact BASE/CANDIDATE source worktrees verified |
| MindIE-SD | `3.0.0` |
| PyAV | `18.0.0` |
| ffmpeg / ffprobe | `4.4.2` / `4.4.2` |

The provenance checks verified that `vllm_omni` imports resolved from the exact
BASE and CANDIDATE source worktrees. The checks accounted for shared editable
distribution metadata and recorded vLLM and vLLM-Ascend provenance separately.

The tables below report measured medians with observed ranges. MP4 reduction is
derived as `(BASE - candidate) / BASE`; the MP4 timer is the direct metric for
this CPU response-conversion change.

| Duration | BASE MP4 (ms) | Candidate MP4 (ms) | Derived reduction |
| ---: | ---: | ---: | ---: |
| 5 s | 4943.92 [4809.49, 5113.98] | 3547.37 [2033.84, 3658.29] | 28.248% |
| 8.7 s | 11456.80 [7834.04, 11647.05] | 3158.92 [3086.81, 5996.05] | 72.428% |
| 15 s | 14367.83 [13426.50, 14546.93] | 5806.33 [5690.51, 10227.57] | 59.588% |

| Duration | BASE stage 0 (s) | Candidate stage 0 (s) |
| ---: | ---: | ---: |
| 5 s | 161.755 [161.057, 162.508] | 157.684 [157.018, 158.686] |
| 8.7 s | 379.430 [353.087, 390.267] | 352.310 [351.838, 359.633] |
| 15 s | 933.645 [902.745, 938.495] | 943.822 [940.118, 960.199] |

| Duration | BASE server E2E (s) | Candidate server E2E (s) |
| ---: | ---: | ---: |
| 5 s | 166.571 [166.007, 167.629] | 160.725 [160.572, 161.349] |
| 8.7 s | 391.083 [364.551, 398.107] | 355.403 [355.003, 365.634] |
| 15 s | 947.077 [917.298, 952.869] | 954.055 [945.930, 965.896] |

Externally observed client-wall E2E was measured outside the server process:

| Duration | BASE externally observed client-wall E2E (s) | Candidate externally observed client-wall E2E (s) |
| ---: | ---: | ---: |
| 5 s | 167.870 [167.322, 168.933] | 162.042 [161.892, 162.664] |
| 8.7 s | 392.409 [365.875, 399.420] | 356.708 [356.317, 366.954] |
| 15 s | 948.404 [918.628, 954.196] | 955.393 [947.237, 967.225] |

Stage 0, server E2E, and externally observed client-wall E2E differences are
not causally attributed to PR2. Only CPU response conversion changed; three
samples show substantial accelerator-side variation. The MP4 timer is the
direct metric.

| Duration | BASE true denoise (s) | Candidate true denoise (s) |
| ---: | ---: | ---: |
| 5 s | 148 [147, 148] | 145 [144, 147] |
| 8.7 s | 332 [331, 369] | 336 [332, 340] |
| 15 s | 894 [869, 900] | 910 [893, 913] |

True denoise elapsed values above are measured with 49 DiT forwards per
request. All-device sampled peak HBM was measured at a 1-second interval; the
peaks are lower bounds:

| Duration | BASE peak HBM (MB) | Candidate peak HBM (MB) |
| ---: | ---: | ---: |
| 5 s | 29794 [29792, 29805] | 31513 [29804, 33220] |
| 8.7 s | 31892 [31771, 31905] | 31770 [31672, 31892] |
| 15 s | 35433 [35433, 35447] | 35433 [35433, 35443] |

All 18 formal MP4s were byte-identical across repeats and between BASE and
candidate for each fixed duration. All 24 total requests (6 warmups plus 18
formal) passed the HTTP, media, and request contracts, including H.264
1344x768 at 24 fps, AAC stereo at 32 kHz, and full timestamp/decode checks.
There was no request-window fatal error, OOM, collective timeout, NaN/Inf,
fallback, or rank failure. The stop gate was clean: the port and processes were
free, and all eight cards were healthy and idle.

The frozen raw-regex audit returned exactly `FINAL_AUDIT=FAIL failures=2`. A
later contextual review classified the two full-log gates as false positives:
each side had 9 startup INFO `NaN-clamp` lines and 32 post-SIGTERM cleanup
`Traceback` markers. The markers occurred after the explicit SIGTERM shutdown
trigger from CANN repository-manager cleanup and were not request failures.

This evidence is limited to fixed Laser configuration, one host, and one
request at a time. It covers no concurrency or sustained recovery, no
`fl2va`/`Ref2VA` or other models/platforms, and supports no general
recommendation beyond this tested H3/A2 setup. The run has no trace-confirmed
Laser operator. The CPU benchmark and worker-boundary scan above are separate
from this E2E evidence.

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
