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

This section records measured CPU response-encoding behavior for MiniMax-H3
FL2VA/t2va on one Atlas A2 host. The implementation is capability-based and
has no model whitelist; other models, platforms, tasks, shapes, durations, and
concurrency levels require separate validation.

Non-streaming MP4 responses use direct-planar conversion only when frame shape,
dtype, and RGB channel-plane contiguity are compatible. The worker option applies
only to that compatible direct-planar conversion; it does not change the legacy
fallback or streaming fMP4 path.

#### Service option and logs

Configure the direct-planar path at `vllm serve` startup with:

```text
--video-response-frame-conversion-workers N
```

The service states are exact: omitted is the default exact PR1 direct-planar
serial iterator, not numeric `1`; explicit `1` is PR2 configured serial
without a pool; explicit `N >= 2` is a bounded FIFO pool.
When supplied, `--video-response-frame-conversion-workers N` requires an
explicit positive integer; there is no fixed public maximum. For F frames,
`W_eff = min(N, F)` and F is the structural upper bound for useful workers;
scheduling, CPU, memory bandwidth, codec, per-worker memory, and concurrency
can impose lower practical limits.

Pooled conversion submits at most `2 * W_eff` frame futures and yields
results in FIFO order. Startup INFO logs `baseline_unconfigured`,
`configured_serial`, or `configured_parallel`, the requested value,
`direct_planar` scope, and the effective-worker rule. Each request logs
`selected_path`, `frame_conversion_mode`, `requested_frame_conversion_workers`,
and `effective_frame_conversion_workers`; omitted is logged as `unset`.
The legacy fallback reports effective workers `0`.

#### Memory model

Each pooled worker reuses one thread-local single-channel scratch buffer. The
consumer still holds the previous frame while fetching the next one, so the
derived conservative per-request capacity is:

```text
min(F, 2 * W_eff + 2) * PyAV_frame_bytes
  + W_eff * single_channel_scratch_bytes
```

For explicitly configured `W_eff = 1`, conversion is serial without a thread
pool, and up to `min(F, 2)` converted frames plus one scratch buffer are used. For
the validated 124-frame, 1344x768 `gbrp` input, one PyAV frame is 3,096,576
bytes and one float32 single-channel scratch buffer is 4,128,768 bytes:

| Configured workers | Pending futures | Known frame slots | Derived capacity |
| ---: | ---: | ---: | ---: |
| `8` | `16` | `18` | `84.65625 MiB` |
| `16` | `32` | `34` | `163.40625 MiB` |
| `32` | `64` | `66` | `320.90625 MiB` |

These are conservative known capacities, not total RSS; they exclude input,
audio, futures, executor objects, libx264 buffers, allocator overhead, and
other process state. Concurrent requests multiply these per-request resources.

#### Atlas A2 service placement

For the measured scope, `8 workers` is the candidate setting. Launch normally
and unbound with this foreground service command:

```bash
export MODEL_ROOT=/path/to/MiniMax-H3
export MODEL="$MODEL_ROOT/FL2VA"
export PORT=9098
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export PYTHONDONTWRITEBYTECODE=1
export MINDIE_SD_FA_TYPE=ascend_laser_attention

vllm serve "$MODEL" \
  --omni --host 0.0.0.0 --port "$PORT" --trust-remote-code \
  --num-gpus 8 --usp 8 --ring 1 --text-encoder-tp-size 8 \
  --enable-distributed-layerwise-offload --vae-parallel-mode tile \
  --vae-use-tiling --vae-patch-parallel-size 8 \
  --diffusion-attention-backend FLASH_ATTN \
  --video-response-frame-conversion-workers 8
```

From another shell, after `/health` returns `200` and all `8` diffusion
workers are ready, identify the API PID and run this separate block:

```bash
read -r -p "API PID: " API_PID
taskset -apc 72-95 "$API_PID"
```

Apply this only to the API PID and its existing threads. Keep the eight
diffusion workers unbound with their PID set and full affinities unchanged.
Retain the default memory policy; do not add `numactl` or `membind`.

#### CPU validation

The three-state compatibility matrix used `taskset` CPU set `72-95` plus
`numactl --cpunodebind=3 --membind=3`. It ran one warmup and `n=10`
pooled formal samples per configuration. This is diagnostic compatibility
evidence, not CPU-only/default-memory evidence and not accepted production
placement.

| Configuration | MP4 wall median (ms) | Process CPU median (ms) | Peak RSS median (KiB) |
| --- | ---: | ---: | ---: |
| `PR1 serial` | `4786.615` | `7961.443` | `3152122` |
| `PR2 option omitted` | `4756.864` | `7962.162` | `3151614` |
| `PR2 explicit 1` | `4822.763` | `8001.731` | `3151610` |

The omitted PR2 state changed wall time by `-0.622%` versus PR1; all three
states produced valid byte-identical media. The later accepted production
placement matrix used only CPU-only `taskset` CPU set `72-95`, default
memory, and no `numactl`/`membind`. It measured `n=10` formal samples:

| Configured workers | MP4 wall median [min, max] (ms) | Process CPU median (ms) | Peak RSS median (KiB) |
| ---: | ---: | ---: | ---: |
| `1 worker` | `4820.797` [4745.619, 4835.065] | `8011.603` | `3153678` |
| `8 workers` | `1865.855` [1836.359, 1953.360] | `9456.770` | `3278242` |

`8 workers` reduced median MP4 encoding wall time by `61.296%` versus
`1 worker`. An independent `8/16/32` scan found no stable wall gain above
`8 workers` while resources rose: forward/reverse medians were
`2854.683/3354.612 ms`, `2855.986/3498.202 ms`, and
`3058.166/3482.307 ms`; corresponding forward/reverse RSS medians were
`3293004/3297688 KiB`, `3462104/3454060 KiB`, and
`3750708/3746896 KiB`. The codec decision retains libx264 `threads=0`;
explicit codec threads `8`, `16`, and `24` showed no stable wall gain of
at least `5%`, and `threads=8` changed the bitstream hash.

#### Final Atlas A2 service comparison

The final comparison used `BASE 3d035bfa190e303f53d72e3baa10885f60abe682`
and tested `CANDIDATE 3e24ea5ba0a498f2f9feab573104b66ddf8dbf55`. It used
`/v1/videos/sync`, MiniMax-H3 `FL2VA/t2va`, one request at a time,
`seed 1101`, `1344x768@24 FPS`, and this exact prompt:

> In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch
> in the snow while the creature breathes and softly snorts.

The protocol used `50 requested scheduler points / 49 DiT forwards`,
Laser-configured attention, libx264 `preset=ultrafast, threads=0`, one
discarded 5-second warmup per service, then `5/8.7/15`-second formal
requests in order, three per duration. BASE omitted the worker flag; CANDIDATE
used `8 workers`. No request was retried.

Recorded hardware/software: `192-CPU Kunpeng 920`, `8x910B4-1`, driver
`25.5.2`, firmware `7.8.0.7.220`, CANN
`9.0.1 (V100R001C10SPC002B220)`, Python `3.12.13`, PyTorch
`2.10.0+cpu`, `torch_npu 2.10.0.post2`, vLLM `0.26.0+empty`,
vLLM-Ascend `0.19.1rc2.dev1251+g905bbf372`, and installed vLLM-Omni
distribution metadata `0.26.1.dev138+g596c16a55.npu`. Separate import
provenance resolved the tested source trees to the exact BASE/CANDIDATE
commits above. MindIE-SD `3.0.0`, PyAV `18.0.0`, and ffmpeg/ffprobe
`4.4.2`.

Response evidence records `x-request-id`, `x-model`,
`x-inference-time-s`, `x-stage-durations`, `x-peak-memory-mb`,
`content-type`, and `content-length`. Startup/request logs record service
state, selected path, conversion mode, requested/effective workers, frame
count, dtype, shape, FPS, audio state, and sample rate. `ffprobe` and full
`ffmpeg` decode verify H.264, AAC stereo at 32 kHz, 24 FPS, dimensions,
timestamps, and frame counts.

Only the MP4 timer is reported as the optimization benefit:

| Requested duration | BASE MP4 median [range] (ms) | CANDIDATE MP4 median [range] (ms) | Derived reduction |
| ---: | ---: | ---: | ---: |
| `5 s` | `4826.50` [4696.01, 4831.13] | `1916.88` [1914.95, 1919.35] | `60.284%` |
| `8.7 s` | `7869.04` [7797.74, 7939.64] | `3184.63` [3167.23, 3190.22] | `59.530%` |
| `15 s` | `13463.56` [13445.20, 13648.02] | `5443.59` [5436.20, 5444.34] | `59.568%` |

| Side | NPU0 | NPU1 | NPU2 | NPU3 | NPU4 | NPU5 | NPU6 | NPU7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `BASE` | `31744` | `35817` | `35816` | `37468` | `39977` | `43920` | `40320` | `39545` |
| `CANDIDATE` | `31529` | `35351` | `35352` | `35352` | `35418` | `35416` | `35412` | `35457` |

The HBM values are maxima from unchanged `hbm-peak-by-device.txt` files,
sampled at 1-second intervals across each side's nine formal requests; they
are lower bounds. Video frame counts are:

| Duration | Video frames |
| --- | ---: |
| `5 s` | `124` |
| `8.7 s` | `209` |
| `15 s` | `362` |

Hashes matched within repeats and across BASE/CANDIDATE for each duration.
All `20` total requests (two warmups and `18` formal) passed
HTTP/media/full-decode/timestamp checks; CANDIDATE logged
`direct_planar configured_parallel 8/8`, all request-level placement and
worker gates passed, and cleanup left `port 9098` free, no service
processes, and all eight NPUs healthy and idle.
No OOM, collective timeout, NaN/Inf, fallback, rank failure, or request
retry occurred.

#### Recommendation

Use `8 workers` only for the validated MiniMax-H3 `FL2VA/t2va` on Atlas
A2 with `192-CPU Kunpeng 920`, `8x910B4-1`, Laser-configured attention,
`1344x768@24 FPS`, and one request at a time. The capability is generic but
untested elsewhere; the default remains omitted.

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
