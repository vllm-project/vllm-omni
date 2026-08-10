# MiniMax-H3 on DGX Spark (GB10)

This recipe uses online FP8 weight quantization, tiled VAE decode, and a single
resident partition. GB10 is a unified-memory platform, so unlike the discrete-GPU
recipes it uses **no offload of any kind** — see the capacity note below.

## Capacity requirements

| Resource | DGX Spark (GB10) |
| --- | ---: |
| Unified memory | 128 GB LPDDR5X (~121 GiB usable) |
| GPUs | 1 |
| Checkpoint storage | 135 GiB per partition |
| Measured peak allocator high-water (FP8) | 97.7 GiB |

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

With ~97.7 GiB of the pool held by the allocator at peak, there is roughly 23 GiB
of headroom for the OS, page cache, and any other process on the box. Treat this
platform as single-tenant while a request is in flight.

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

## Measured latency

Single GB10, FP8, eager, cuDNN attention, tiled VAE, one request at a time:

| Workload | Steps | Text encode | Denoise | Per step | VAE decode | MP4 mux | Client E2E | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2VA, 4 s, smoke shape | 10 | — | — | 8.68 s | — | 0.49 s | 95.9 s | — |
| T2VA, 8 s, 960x576 | 50 | 0.25 s | 2088 s | 42.61 s | 70 s | 4.84 s | 2169.4 s (36 min 9 s) | 97.7 GiB |

How these were measured:

- Stage times come from `--enable-diffusion-pipeline-profiler`, read from the
  `X-Stage-Durations` response header of `/v1/videos/sync`.
- Peak memory is the `X-Peak-Memory-MB` header, which reports the CUDA allocator
  high-water mark (`torch.cuda.max_memory_reserved`) for the request. It excludes
  the CUDA context and non-PyTorch allocations, so true pool usage is somewhat
  higher. `nvidia-smi --query-gpu=memory.used` is not usable on GB10 — unified
  memory has no discrete VRAM to report and the field returns `[N/A]`.
- Text encode and VAE decode were measured on a separate 10-step run at the same
  960x576 / 8 s geometry. Both are independent of step count, so they transfer to
  the 50-step row unchanged.
- Stage times do not sum exactly to end-to-end. Queueing, result transfer, and
  engine bookkeeping sit outside the profiled stages.

Denoise dominates at 96% of end-to-end, so step count and geometry are the only
knobs that materially change the wall clock on this platform.

Repeated identical 10-step requests varied by up to 23% in denoise time
(397 s to 490 s) on a machine that had already been under sustained load. Treat
these as single-run measurements on one machine, not a throughput guarantee, and
expect the slower end of that range for back-to-back jobs.

Set `VLLM_OMNI_VIDEO_SYNC_TIMEOUT` well above the expected request time — the
default 1800 s is shorter than a 50-step run on this platform.

## Known limitations

- One GPU means no Ulysses, ring, TP, or VAE patch parallelism. Pass
  `--usp 1 --ring 1 --vae-patch-parallel-size 1` explicitly if a profile or
  script would otherwise supply higher degrees.
- `--enforce-eager` is used because regional compile adds startup time without a
  parallelism win on a single rank.
- `t2va` rejects requests without an explicit `aspect_ratio`. When `width` and
  `height` are both supplied they take precedence and `aspect_ratio` only has to
  name one of the supported ratios.
- Sustained load degrades throughput. See the variance note above.
- Only one partition is resident. For Ref2VA, stop the FL2VA server and restart
  the same command with `/path/to/MiniMax-H3/Ref2VA`. Ref2VA reference video
  count and prompt length increase activation memory; begin with one request at
  a time and keep the short edge at 576.
- Online FP8 is incompatible with layerwise offload — the offload path produces a
  weight stride the Cutlass FP8 kernel rejects. This is not a practical
  restriction here since offload is unusable on GB10 anyway.
