# MiniMax-H3 on a single NPU 950PR

This recipe runs MiniMax-H3 with online INT8 quantization on one NPU 950PR
NPU (128 GB HiBL 1.0 HBM). It covers the single-card T2VA configuration at
1024x576. For the eight-card Atlas 800I A3 BF16 route at 768P, see
[MiniMax-H3-NPU.md](MiniMax-H3-NPU.md).

## Capacity requirements

| Resource | Requirement |
| --- | ---: |
| NPU | 1x NPU 950PR |
| NPU HBM | 128 GiB (131,072 MiB reported by `npu-smi`) |
| Observed HBM high-water mark | 118,442 MiB (90.4% of capacity) |
| Checkpoint storage | 135 GiB per partition, local disk strongly preferred |
| Container shared memory | 8 GiB, or a writable local filesystem for spill |

The observed high-water mark leaves about 12.3 GiB of headroom, so this card is
sized for exactly one resident task partition at 1024x576. Do not expect a
larger output shape or concurrency greater than one to fit.

Because a single 950PR holds the whole partition in HBM, this route does not
need layerwise offload, and the 200 GiB system-RAM floor carried by the 72 GiB
GPU recipes does not apply. The validated run completed inside a container
limited to 32 GiB of RAM. Passing `--enable-layerwise-offload` on this card is
actively harmful: it stages weights in non-reclaimable host memory and the
container's OOM killer terminates the server with exit code -9.


## Environment

- Host architecture: x86_64
-  driver: 25.7.rc1.6 (hal 7.35.23)
-  firmware: 9.0.0.105.229
- CANN toolkit: 9.1.0 (`/usr/local//cann-9.1.0`)
- npu-smi: 25.7.rc1.6
- Python: 3.12.13
- PyTorch: 2.10.0+cpu
- torch_npu: 2.10.0.post2
- vLLM: 0.26.0
- vLLM-Omni: 0.26.1.dev103+g584d78c67.npu (commit `584d78c6`)

Install vLLM-Omni from a checkout with MiniMax-H3 support:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Start a server

```bash
export MODEL=/path/to/MiniMax-H3/FL2VA
export PORT=8000
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 1 \
  --tensor-parallel-size 1 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 1 \
  --vae-patch-parallel-size 1 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --init-timeout 14400 \
  --stage-init-timeout 14400 \
  --quantization int8
```

Do not pass `--diffusion-attention-backend CUDNN_ATTN`. cuDNN attention has no
NPU implementation and fails at the first denoise step. Leave the backend
unset: it resolves to `TORCH_SDPA`, or to `FLASH_ATTN` when MindIE-SD is
installed (see [MiniMax-H3-NPU.md § Environment](MiniMax-H3-NPU.md#environment)).

H3 is CFG-distilled, so `--cfg-parallel-size` must remain 1.

## Validated evidence

Measured on one NPU 950PR with the server command above plus
`--enable-diffusion-pipeline-profiler`, generating a 5 s 1024x576 T2VA clip from
the request in [§ T2VA request example](#t2va-request-example). MiniMax-H3
requested 60 denoise steps and executed 59 denoise updates, so per-step latency
is `denoise / 59`. `ffprobe` confirms every returned MP4 is 1024x576, 24 fps,
124 video frames, 5.175 s, with a 32 kHz audio track of 165,600 samples.

Five requests were issued back to back with no idle time between them. The card
is measurably faster on the first request than in steady state, so both regimes
are reported.

| Stage | First request, cold card | Thermal steady state |
| --- | ---: | ---: |
| End-to-end request | 462.13 s | 506.38 s |
| Text encode | 0.55 s | 0.05 s |
| Denoise (59 updates) | 429.29 s | 471.74 s |
| Per denoise update | 7,276 ms | 7,996 ms |
| VAE decode | 8.43 s | 8.14 s |
| Worker-to-server handoff | 24 s | 27 s |
| MP4 muxing | 0.57 s | 0.50 s |
| Server-reported `denoise_step_latency_ms` | 7,702 ms | 8,440 ms |
| Peak HBM (`npu-smi`) | 118,442 MiB | 118,442 MiB |
| Average NPU power | 566.9 W | 556.9 W |
| Peak NPU temperature | 104 C | 104 C |

Sample counts differ per row. The cold column is a single request. In the steady
column, text encode, denoise and VAE decode are means over the four subsequent
requests — denoise spanned 470.37 to 473.21 s, a 0.60% spread — while
end-to-end, handoff and muxing are means over the two of those four that
survived the handoff timeout described above.

**Report the steady-state column.** The cold card completes denoise 9.9% faster
while drawing only 1.8% more average power, and both regimes reach the same
104 C ceiling, so the first request is buying a short window of higher clocks
before the die saturates. Any benchmark that issues a single request against an
idle card will overstate throughput by about 10%.

**Do not read `denoise_step_latency_ms` as a per-step time.** The server divides
the entire stage wall time by the 60 *requested* steps, so it absorbs text
encoding, VAE decode and the 837 MiB handoff. All three successful requests
satisfy `denoise_step_latency_ms == e2e_stage_wall_time_ms / 60` exactly. The
per-update figure in the table is `diffuse / 59` from the pipeline profiler.

Denoise accounts for 93% of the steady-state request and the handoff accounts
for 5.3%, so keeping the transfer in shared memory should bring end-to-end down
to roughly 480 s.

Peak HBM is sampled externally with `npu-smi` at a 3 s interval. It is a
reserved high-water mark rather than live allocation: it reads 111,272 MiB after
weight loading and before the first request, and does not fall back after
requests complete. Re-measure for longer outputs, a different output shape, or
concurrency greater than one.

### Startup

Weight loading took **31 min 50 s** on the validated host (first log line
08:04:59, `Application startup complete` 08:36:49), with the checkpoint on a
shared GlusterFS network mount. This is why the server command sets
`--init-timeout 14400` and `--stage-init-timeout 14400`; the stock 600 s and
1800 s timeouts both abort mid-load with `TimeoutError` and exit code 143.
Reduce both timeouts when the partition is staged on local disk.

## T2VA request example

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"

curl -sS --max-time 1800 -X POST "${API_URL}" \
  -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
  -F 'width=1024' \
  -F 'height=576' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=60' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

## Known limitations

- INT8 online quantization is validated for T2VA on this card. Use BF16 for
  FL2VA and Ref2VA, or re-measure before relying on it.
- Single-card serving loads one task partition at a time. For Ref2VA, stop the
  server and restart it against the `Ref2VA` directory.
- Sustained load drives the die to 104 C and costs about 10% of denoise
  throughput relative to a cold card. Warm the server with at least one full
  request before measuring, and treat steady state as the reportable number.
- Requests can fail during result handoff on hosts where the IPC spill path is
  slow. See [§ Result handoff and shared
  memory](#result-handoff-and-shared-memory).
- Loading the checkpoint from a network mount takes over 30 minutes and forces
  very large init timeouts. Stage the partition on local disk when possible.
- The configuration measured here is 1024x576 at 60 steps, which differs from
  the 1344x768 at 50 steps used by the GPU recipes in this directory. Denoise
  cost scales superlinearly with token count, so the numbers are not directly
  comparable across recipes.
- The image ships a `triton` package whose  backend is not built
  (`No module named 'triton._C.libtriton.'`). vLLM logs this as an error
  at startup and disables Triton; the diffusion path does not need it and the
  run completes normally.

## Additional resources

- [MiniMax-H3.md](MiniMax-H3.md) — full GPU guide
- [MiniMax-H3-NPU.md](MiniMax-H3-NPU.md) — eight-card Atlas 800I A3 BF16 guide
- [Int8 quantization](../../docs/user_guide/quantization/int8.md)
- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
