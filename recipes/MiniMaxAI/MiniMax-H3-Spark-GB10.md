# MiniMax-H3 on DGX Spark (GB10)

This recipe uses online FP8 weight quantization, tiled VAE decode, and a single
resident partition. GB10 is a unified-memory platform, so unlike the discrete-GPU
recipes it uses **no offload of any kind** — see the capacity note below.

Validated on:

- Host: DGX Spark (GB10), aarch64
- GPUs: 1 (unified memory)
- Driver: <FILL_DRIVER_VERSION>
- vLLM: 0.26.0
- vLLM-Omni: `main` at `e1aa6eae75c460cd1893bc320546e81e66973831`
- Workloads: T2VA and Ref2VA, 960x576, `duration=8.0`, 50 steps, `flow_shift=12`,
  `seed=1101`


## Capacity requirements

| Resource | DGX Spark (GB10) |
| --- | ---: |
| Unified memory | 128 GB LPDDR5X (~121 GiB usable) |
| GPUs | 1 |
| Checkpoint storage | 135 GiB per partition |
| Measured peak allocator high-water, T2VA (FP8) | 97.7 GiB |
| Measured peak allocator high-water, Ref2VA (FP8) | 102.8 GiB |

`FL2VA` and `Ref2VA` are separate 135 GiB checkpoint partitions. Start one server
at a time.

On GB10 the CPU and GPU share one physical memory pool, which changes the capacity
math relative to the RTX and datacenter recipes:

- **Do not use `--enable-distributed-layerwise-offload`.** DLO stages rank-local
  weights into pinned host memory, which on GB10 is the same pool the weights are
  already in. Peak usage roughly doubles and the process is killed by the OOM
  killer (`Exit code: -9`) shortly after `Enabling offloader backend`.
- **Do not use `--enable-cpu-offload`.** Moving weights from "VRAM" to "host RAM"
  frees nothing here.
- **`--quantization fp8` is mandatory.** A BF16 partition is 135 GiB and does not
  fit in 121 GiB. Online FP8 quantizes the DiT only (62 GiB to ~31 GiB); the
  Qwen3-VL text encoder and both VAEs stay BF16.

With ~97.7 GiB of the pool held by the allocator at peak, T2VA leaves roughly
23 GiB of headroom for the OS, page cache, and any other process on the box.
Ref2VA is tighter on both counts. The allocator high-water rises to 102.8 GiB,
and the allocator figure understates true pool usage.

Sampling `/proc/meminfo` at 1 Hz across a 50-step Ref2VA request on a host with
`MemTotal: 127600524 kB` (121.7 GiB), `MemAvailable` started at 8.1 GiB, fell to
a trough of **7.4 GiB**, and recovered to 8.9 GiB. At the trough the pool held
roughly **114.3 GiB** — 11.4 GiB more than the 102.8 GiB the allocator reports,
which is the CUDA context, non-PyTorch allocations, and page cache the header
does not count.

Treat this platform as strictly single-tenant for Ref2VA. There is no room for a
second process, and adding reference images, reference videos, or a larger output
shape will run into the OOM killer.

## One GB10: 960x576, 8 seconds

```bash
CUDA_VISIBLE_DEVICES=0 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --init-timeout 3600 \
  --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
  --usp 1 --ring 1 --vae-patch-parallel-size 1 \
  --vae-parallel-mode tile --vae-use-tiling \
  --quantization fp8 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

Wait for `Application startup complete`, then run a short smoke test before
committing to a full 50-step request.

`t2va` requires an explicit `aspect_ratio` even when `width` and `height` are
supplied; omitting it fails the request with
`OmniClientError: t2va requires an explicit aspect_ratio`.

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8000/v1/videos/sync \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F 'width=960' \
  -F 'height=576' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=10' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":4.0,"audio_flow_shift":3.0}' \
  -o h3-gb10-smoke.mp4
```

The full-quality request uses 50 steps:

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8000/v1/videos/sync \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F 'width=960' \
  -F 'height=576' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8.0,"audio_flow_shift":3.0}' \
  -o h3-gb10-t2va.mp4

ffprobe -v error -show_entries \
  stream=index,codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of json h3-gb10-t2va.mp4
```

Both requests return `200 OK` with an MP4 body: H.264 video at the requested
geometry plus a 32 kHz stereo AAC track.

## One GB10: Ref2VA, 960x576, 8 seconds

`FL2VA` and `Ref2VA` are separate 135 GiB partitions and only one fits at a time.
Stop the FL2VA server before starting this one.

```bash
CUDA_VISIBLE_DEVICES=0 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve /path/to/MiniMax-H3/Ref2VA \
  --omni --trust-remote-code --host 0.0.0.0 --port 8000 \
  --init-timeout 3600 \
  --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
  --usp 1 --ring 1 --vae-patch-parallel-size 1 \
  --vae-parallel-mode tile --vae-use-tiling \
  --quantization fp8 --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

The 7200 s timeout used for T2VA is too short here: a 50-step Ref2VA request
takes over an hour on this platform. Use 14400 s.

Unlike `t2va`, `ref2va` defaults to a 16:9 output ratio, so `aspect_ratio` can be
omitted when `width` and `height` are supplied.

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8000/v1/videos/sync \
  -D ref2va-headers.txt \
  -w 'client_e2e_s=%{time_total}\n' \
  --max-time 14400 \
  -F 'input_reference=@/path/to/reference.jpg;type=image/jpeg' \
  -F 'prompt=A hand rests on a wooden table in warm evening light, then slowly opens and turns toward the camera, with quiet room ambience.' \
  -F 'width=960' \
  -F 'height=576' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"ref2va","duration":8.0,"audio_flow_shift":3.0}' \
  -o h3-gb10-ref2va.mp4
```

The response is `200 OK` with an MP4 body: H.264 at the requested geometry plus a
32 kHz stereo AAC track. `ffprobe` reports 8.032 s for `duration=8.0`.

## Measured latency

Single GB10, FP8, eager, cuDNN attention, tiled VAE, one request at a time:

| Workload | Steps | Text encode | Denoise | Per step | VAE decode | MP4 mux | Client E2E | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2VA, 4 s, smoke shape | 10 | — | — | 8.68 s | — | 0.49 s | 95.9 s | — |
| T2VA, 8 s, 960x576 | 50 | 0.25 s | 2088 s | 42.61 s | 70 s | 4.84 s | 2169.4 s (36 min 9 s) | 97.7 GiB |
| Ref2VA, 8 s, 960x576, one image (3 runs) | 10 | 8.43-10.94 s | 665.5-748.8 s | 66.55-74.88 s | 67.9-69.8 s | 3.63 s | 770.3-861.1 s | 102.8 GiB |
| Ref2VA, 8 s, 960x576, one image | 50 | 9.66 s | 4039.6 s | 80.79 s | 69.4 s | 3.63 s | 4157.0 s (69 min 17 s) | 102.8 GiB |

How these were measured:

- Stage times come from `--enable-diffusion-pipeline-profiler`, read from the
  `X-Stage-Durations` response header of `/v1/videos/sync`. The keys are
  `MiniMaxH3Pipeline.encode_prompt` (text encode),
  `MiniMaxH3Pipeline.diffuse` (denoise), and `MiniMaxH3Pipeline.decode`
  (VAE decode).
- MP4 mux is not a header field. It is the
  `Video response encoding (MP4 bytes)` line in the server log.
- Peak memory is the `X-Peak-Memory-MB` header, which reports the CUDA allocator
  high-water mark (`torch.cuda.max_memory_reserved`) for the request. It excludes
  the CUDA context and non-PyTorch allocations, so true pool usage is somewhat
  higher. `nvidia-smi --query-gpu=memory.used` is not usable on GB10 — unified
  memory has no discrete VRAM to report and the field returns `[N/A]`. See the
  capacity section for a `/proc/meminfo` cross-check.
- For T2VA, text encode and VAE decode were measured on a separate 10-step run at
  the same 960x576 / 8 s geometry. Both are independent of step count, so they
  transfer to the 50-step row unchanged. The Ref2VA MP4 mux figure transfers the
  same way, from the 10-step run at identical output geometry and duration.
- Ref2VA peak memory is byte-identical between the 10-step and 50-step runs
  (`X-Peak-Memory-MB: 105318.000` in both), confirming that peak is set by output
  geometry and reference input rather than step count.
- Stage times do not sum exactly to end-to-end. Queueing, result transfer, and
  engine bookkeeping sit outside the profiled stages.

Denoise dominates at 96% of end-to-end, so step count and geometry are the only
knobs that materially change the wall clock on this platform.

Ref2VA costs 1.92x the wall clock of T2VA at the same geometry and step count:
4157.0 s versus 2169.4 s. Text encoding shows the most dramatic ratio — 9.66 s
versus 0.25 s, 39x — but at 0.2% of end-to-end it is not what makes Ref2VA slow.
The cost is in denoise. A single reference image expands the Qwen3-VL
presentation to 6883 tokens (logged as `MiniMax H3 ref2va Qwen presentation:
6883 tokens, 1 reference images`), and those tokens sit in the attention context
of every denoise step. Per-step time rises from 42.61 s to 80.79 s, and that
1.90x multiplies across all 50 steps. VAE decode is unchanged at ~69 s, as
expected for an identical output shape.

Repeated identical 10-step T2VA requests varied by up to 23% in denoise time
(397 s to 490 s) on a machine that had already been under sustained load. Treat
these as single-run measurements on one machine, not a throughput guarantee, and
expect the slower end of that range for back-to-back jobs.

Ref2VA per-step time is load-dependent in the same way. On an idle machine the
first 10-step request ran at 66.55 s per step. The 50-step request that followed
occupied the box for 69 consecutive minutes and averaged 80.79 s per step. Two
further 10-step requests taken immediately afterwards settled at 74.88 s and
74.31 s — 12.5% slower than the idle run, and within 0.8% of each other, so this
is a stable thermal state rather than measurement noise.

Plan for 81 s per step when sizing a full 50-step Ref2VA job; 66 s per step is
only reachable on a cold machine and does not hold for the length of one request.

Ref2VA text encode converged downward across the four runs (10.94 s, 9.66 s,
8.52 s, 8.43 s) as first-touch costs fell away; treat ~8.5 s as steady state.
VAE decode was stable at 67.9-69.8 s throughout.

Set `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` well above the expected request time — the
default 1800 s is shorter than a 50-step run on this platform, and 7200 s is
shorter than a 50-step Ref2VA run.

## Known limitations

- One GPU means no Ulysses, ring, TP, or VAE patch parallelism. Pass
  `--usp 1 --ring 1 --vae-patch-parallel-size 1` explicitly if a profile or
  script would otherwise supply higher degrees.
- `--enforce-eager` is used because regional compile adds startup time without a
  parallelism win on a single rank.
- `t2va` rejects requests without an explicit `aspect_ratio`. When `width` and
  `height` are both supplied they take precedence and `aspect_ratio` only has to
  name one of the supported ratios.
- Sustained load degrades throughput. See the variance notes above.
- Only one partition is resident. For Ref2VA, stop the FL2VA server and restart
  with `/path/to/MiniMax-H3/Ref2VA`. Ref2VA with one reference image is measured
  above; reference videos and additional images push the token count higher and
  were not measured. With `MemAvailable` bottoming out at 7.4 GiB during a
  single-image request, there is very little room to grow — re-measure before
  adding references or raising the output shape.
- Ref2VA requires a vLLM-Omni build newer than `v0.26.0`. See the version note
  under **Validated on**: FP8 weight loading and image-only Ref2VA are both
  broken on `release/v0.26.0`.suggestion is  `v0.26.1`.
- Online FP8 is incompatible with layerwise offload — the offload path produces a
  weight stride the Cutlass FP8 kernel rejects. This is not a practical
  restriction here since offload is unusable on GB10 anyway.
```
