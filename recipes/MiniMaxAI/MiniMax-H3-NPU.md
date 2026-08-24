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

### CPU MP4 response encoding (Atlas A2)

Non-streaming MP4 responses use one public automatic encoder. It checks the
runtime frame shape, common dtype, and RGB channel-plane contiguity for every
request; compatible inputs use direct planar PyAV frames, while unsupported
inputs fall back to the legacy muxer before opening the PyAV container. No CLI
flag, model declaration, or user configuration is required. Streaming fMP4
output remains on its existing incremental path.

Current performance and correctness validation is limited to one Atlas A2 host
with 8x Ascend 910B4-1 NPUs, the `FL2VA`/`t2va` partition, one request at a
time, 1344x768 at 24 fps, and 5, 8.7, and 15 second requests. This optimization
only changes CPU MP4 response encoding; it does not change DiT execution or
stage 0.

#### Event-loop scheduling

Complete non-streaming raw MP4 encoding runs in a dedicated single-worker
executor instead of the API server's asyncio event loop. This covers the
background artifact produced by `/v1/videos`, the raw MP4 response from
`/v1/videos/sync`, and both the direct-planar and legacy fallback routes. The
existing base64 response handler also uses the executor, but it currently has
no HTTP route. Streaming fMP4 output keeps its existing incremental path. The
executor is configured internally at service startup; there is no CLI option
or request parameter.

One response is admitted to native encoding at a time. Additional request
coroutines wait before executor submission, so native encoding work does not
accumulate in the executor queue. Waiting requests can still retain their
generated artifacts, and the design does not impose a global request-memory
bound.

Cancellation or a synchronous endpoint timeout cannot stop work that has
already entered the native codec. That work retains the encoding slot until it
finishes. During service shutdown, background video jobs are cancelled and
awaited before the executor rejects pending submissions, cancels work that has
not started, and waits for the active encode to complete.

The startup log records `execution=dedicated_executor`,
`scope=non_streaming_mp4`, `paths=raw_mp4,base64_handler`, and
`max_active_encodes=1`.
Per-response encoding logs record `queue_inclusive=true` and the execution
mode, so their duration includes time waiting for the slot.

A fixed-input CPU benchmark used a 124-frame H3 dump at 1344x768 with stereo
32 kHz audio, one warmup, and five formal runs per side on CPUs 72-95. Moving
the encode reduced the median maximum event-loop heartbeat gap from 4766.558 ms
to 10.033 ms (-99.79%) and its p99 from 4623.715 ms to 5.160 ms (-99.89%).
The outputs were byte-identical; encode wall time and process CPU time remained
diagnostic rather than acceptance metrics.

**Measured A2 service validation (2026-08-24)** compared base commit
`d150a4fde77d15d466102323a4048b0a8631d74c` with candidate commit
`b10656285122b463a2bb868fdac35e9cb8cf7969`. Both used MiniMax-H3
`FL2VA`/`t2va` on one Atlas A2 host with 8x Ascend 910B4-1 NPUs, the eight-NPU
command above with `MODEL=$MODEL_ROOT/FL2VA`, and
`MINDIE_SD_FA_TYPE=ascend_laser_attention`. The API process was pinned to CPUs
72-95 with the default NUMA memory policy. Each side discarded one 5-second
warmup, then ran three 5-second requests, one at a time, at 1344x768 and 24 fps
with `seed=1101`, `flow_shift=12`, `audio_flow_shift=3.0`, 50 requested steps,
and 49 DiT forwards. The fixed prompt was: "In a snowy blue-purple forest, Ori
carefully walks past a sleeping giant; footsteps crunch in the snow while the
creature breathes and softly snorts."

| Formal median (`n=3`) | Base | Candidate | Change |
| --- | ---: | ---: | ---: |
| Maximum `/health` response during each request | 4.350 s | 0.185 s | -95.74% |
| MP4 encoding | 4726.080 ms | 4816.380 ms | +1.91% |
| Stage 0 | 158465.635 ms | 160077.244 ms | +1.02% |
| Server E2E | 163.138 s | 164.909 s | +1.09% |

All six formal outputs were byte-identical, with SHA-256
`246100788ea4a839a3b0dc1a7a33405dd9b1f6252b6f6889cf8c70682f73c351`,
and passed full decode as 124-frame H.264 1344x768 video at 24 fps with AAC
stereo audio at 32 kHz. No rank failure, OOM, collective timeout, NaN/Inf, or
backend fallback occurred. At 1-second sampling, the per-device formal peak HBM
values for NPU 0-7 were `[28803, 29477, 29477, 32448, 29564, 29563, 29563,
31030]` MB for Base and `[28803, 29475, 30943, 31166, 29564, 29562, 29561,
29563]` MB for the candidate.

The validated stack was driver 25.5.2, firmware 7.8.0.7.220, CANN 9.0.1,
Python 3.12.13, PyTorch 2.10.0+cpu, torch_npu 2.10.0.post2, vLLM
0.26.0+empty, vLLM-Ascend 0.19.1rc2.dev1251+g905bbf372, MindIE-SD 3.0.0,
PyAV 18.0.0, and ffmpeg/ffprobe 4.4.2. The `/health` result measures event-loop
responsiveness. MP4, Stage 0, and E2E changes are diagnostic variation rather
than encoding or generation acceleration. The mechanism is model- and
platform-neutral, but it has not yet been tested on other deployments.

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
| -------- | ------------- |
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
