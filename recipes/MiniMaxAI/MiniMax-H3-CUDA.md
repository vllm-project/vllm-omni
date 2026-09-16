# MiniMax-H3 on NVIDIA CUDA

[Model guide](MiniMax-H3.md) · [Deployment choices](MiniMax-H3.md#choose-a-deployment) · [HTTP API](MiniMax-H3.md#http-api-examples)

This guide contains CUDA deployment configurations and their validation results.
For model inputs, adapters, and requests, use the [model guide](MiniMax-H3.md).
The [RTX 4090](MiniMax-H3-4090.md), [RTX 5090](MiniMax-H3-5090.md),
[RTX PRO 5000](MiniMax-H3-RTX-PRO-5000.md),
[RTX PRO 6000](MiniMax-H3-RTX-PRO-6000.md), and
[DGX Spark](MiniMax-H3-Spark-GB10.md) guides provide device-specific profiles.

## Installation

Complete the model guide's [checkpoint prerequisites](MiniMax-H3.md#prerequisites).

Install vLLM-Omni from the checkout containing MiniMax H3 support:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

The cuDNN configurations in the RTX 4090/5090 recipes do not require FA4.
For the optional `FLASH_ATTN` backend on datacenter Blackwell, install the
FlashAttention-4 extra:

```bash
uv pip install -e '.[fa4]'
```

## Memory and storage requirements

See the model guide for [checkpoint storage](MiniMax-H3.md#checkpoint-storage).
GPU memory and host RAM are separate deployment requirements.

For offloaded deployments, budget host RAM for model weights and pinned
transfer buffers in addition to accelerator memory. The device-specific
recipes give the requirements for their selected topology and workload.

## Four GPUs: throughput-oriented combined service

For a combined service on four high-memory datacenter Blackwell GPUs, use:

- no CPU or layerwise offload;
- Ulysses sequence parallelism degree 4;
- native tiled VAE patch parallelism degree 4;
- regional `torch.compile` for the repeated DiT blocks;
- dense BF16 `TRTLLM_ATTN`, with Ring and TP left at 1.

Both DiTs remain resident in this no-offload configuration. If they do not fit,
use model-level CPU offload.

```bash
vllm serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling
```

Do not add `--enforce-eager` to this performance configuration. The first
request includes regional compilation; warm the server once before measuring
steady-state latency. H3 is CFG-distilled, so `--cfg-parallel-size` must remain
`1`. The H3 VAE supports its native `tile` mode, not
`spatial_shard_height` or `spatial_shard_width`.

## Single GPU: blockwise capacity path

Use ordinary layerwise offload when one GPU cannot keep the Qwen3-VL encoder
and DiT resident together. The component list below streams the active DiT,
the Qwen vision blocks, and the first 50 Qwen text layers. Encoder blocks stay
rank-local; the video/audio VAEs remain resident.

```bash
vllm serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 1 \
  --diffusion-offload-config \
  '{"mode":"layer","components":["dit","text_encoder"]}' \
  --enforce-eager \
  --diffusion-attention-backend FLASH_ATTN
```

This is a capacity profile, not a latency profile: every denoising step streams
DiT blocks over the host link, while encoder blocks are streamed only during
the short conditioning phase. It requires host memory for the complete
checkpoint plus pinned transfer buffers. Re-measure peak HBM on the target
shape; this command does not claim a particular GPU model as validated.

A one-B300 correctness smoke selected only `text_encoder` (keeping the DiT and
VAEs resident) and completed a 384x672, 5-second T2VA request with two denoise
steps. It installed 77 encoder hooks across the vision and text stacks, used a
77,728 MiB worker peak, and measured 2.803 seconds encode, 2.706 seconds
denoise, and 4.503 seconds decode. These reduced-step numbers validate the
execution path; they are not a quality or production-latency benchmark.

To use whole-component behavior, change the config to
`{"mode":"module","components":["dit","text_encoder"]}`. Module
offload swaps the complete encoder and DiT and therefore has a higher
encode-phase peak than encoder blockwise offload.

## Single GPU: low-memory serving

The [RTX 4090](MiniMax-H3-4090.md#single-gpu) and
[RTX 5090](MiniMax-H3-5090.md#single-gpu) recipes use rank-local distributed
layerwise offload with no retained DiT layers. On one worker, no weight
collective is needed. DiT and text-encoder blocks stream from host memory;
the video and audio VAEs move to the device only for their encode/decode
phases. VAE tiling bounds each decode tile's activation footprint. This
placement keeps the model's existing weight precision and sampling schedule.

Select `dit`, `text_encoder`, and `vae`, with DiT
`weight_transfer: rank-local`, as shown in the device recipes. Omitting `vae`
keeps the VAEs resident; explicit rank-local transfer retains the bounded
streaming path even with zero resident DiT layers.

For compatible unquantized TP1 DiT weights, the loader retains checkpoint
mmap views and transfers blocks through bounded pinned buffers. The text
encoder and VAEs still retain host copies. This is a GPU-memory reduction
path, not a validated 32 GiB system-RAM deployment: account for those copies,
startup allocations, transfer buffers, and the OS page cache. Local SSD
storage is preferable when checkpoint pages must be read repeatedly.
See [host-weight loading](../../docs/user_guide/diffusion/offloader/distributed_layerwise_offload.md#host-weight-loading).

After starting either single-GPU server, begin with one 480P T2VA request:

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/v1/videos/sync \
  -F 'prompt=A small boat crosses a calm lake at sunrise, with rippling water and gentle birdsong.' \
  -F 'aspect_ratio=16:9' -F 'width=864' -F 'height=480' \
  -F 'fps=24' -F 'num_inference_steps=50' -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5}' \
  -o h3-t2va.mp4
```

This request is a starting configuration. Single-GPU end-to-end validation,
peak device and host memory, and latency remain unmeasured for these profiles.

## FastH3 VSA

Complete the [checkpoint prerequisites](MiniMax-H3.md#prerequisites) and
install the `vsa` extra using the
[VSA installation guide](../../docs/user_guide/diffusion/attention_backends/fastvideo_vsa.md).
The model's [adapter contract](MiniMax-H3.md#fasth3-adapter) applies.

Download the VSA / Data-Free adapter:

```bash
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  vsa-datafree/adapter_model.safetensors --local-dir ./fasth3
```

Start four workers with pure Ulysses:

```bash
vllm serve MiniMaxAI/MiniMax-H3 --omni --trust-remote-code \
  --task-type fl2va --lora-path ./fasth3/vsa-datafree/adapter_model.safetensors \
  --usp 4 --diffusion-attention-backend FASTVIDEO_VSA
```

This four-GPU example is configuration-only; full-model VSA end-to-end
validation is pending.

Once the server is ready, send a request from another terminal:

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/v1/videos/sync \
  -F 'prompt=A small boat crosses a calm lake at sunrise, with rippling water and gentle birdsong.' \
  -F 'aspect_ratio=16:9' -F 'num_inference_steps=4' \
  -F 'extra_params={"task":"t2va","duration":4.4}' \
  -o fasth3-vsa.mp4
```

## Attention Backends

On supported datacenter Blackwell systems, MiniMax H3 defaults to dense BF16
`TRTLLM_ATTN`; no attention backend flag is required. To select it explicitly,
use:

```bash
--diffusion-attention-backend TRTLLM_ATTN
```

Stable measurements with the [four-GPU profile](#four-gpus-throughput-oriented-combined-service) put dense `TRTLLM_ATTN`
and FA4 within 2% of each other. `TRTLLM_ATTN` remains the datacenter Blackwell
default and enables the optional optimizations below. Confirm the server log
contains `Defaulting to diffusion attention backend TRTLLM_ATTN` before
recording measurements when using the default selection.

FA4 remains available by explicitly selecting the `FLASH_ATTN` backend:

```bash
--diffusion-attention-backend FLASH_ATTN
```

With the optional `[fa4]` dependency installed, `FLASH_ATTN` prefers FA4 on
Blackwell. Confirm the server log contains `Using CuTe FlashAttention-4 on
Blackwell` before recording FA4 measurements.

`TRTLLM_ATTN` additionally offers two **lossy** optimizations for the long main
DiT attention sequence: SAGE attention quantization and Skip-Softmax sparse
attention. Both work under the pure Ulysses parallelism of that profile
(`--usp 4 --ring 1`). The example below enables both:

- SAGE with `fp8_e4m3` Q/K; P and V are always FP8 in this kernel. B200
  additionally supports `int8` Q/K, which preserves accuracy better than FP8.
- Skip-Softmax with a direct `threshold=0.05` (the calibrated
  `target_sparsity` control needs ModelOpt metadata that the official H3
  checkpoint does not include). Together with the cutoff below this is a
  **conservative** setting: a low threshold skips only clearly negligible tiles,
  and a cutoff close to `1.0` leaves a substantial dense prefix. Raise
  `threshold` or lower `disabled_until_timestep` for more speedup once quality
  is verified.
- `disabled_until_timestep=0.97` keeps the early high-noise steps dense. The
  gate compares against the video sigma, which H3's default flow shift of 12
  keeps high for much of the run: at 50 steps, `0.99`, `0.97`, and `0.95`
  leave the first 6, 14, and 19 of 49 denoiser forwards dense. See the
  [Skip-Softmax design](https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/skip_softmax.md#timestep-gating)
  for how the cutoff maps to steps.
- A `per_role` entry that keeps the token refiner, a short attention path,
  dense. A per-role spec does not inherit `quant` or `skip_softmax` from
  `default`.

```bash
--diffusion-attention-config '{
  "default": {
    "backend": "TRTLLM_ATTN",
    "quant": {
      "dtype_qk": "fp8_e4m3",
      "q_block_size": 1,
      "k_block_size": 16
    },
    "skip_softmax": {
      "threshold": 0.05,
      "disabled_until_timestep": 0.97
    }
  },
  "per_role": {
    "minimax_h3.token_refiner": {
      "backend": "TRTLLM_ATTN"
    }
  }
}'
```

Both optimizations trade fidelity for speed and their effects compound.
Compare against dense output on the same prompt and seed before adopting them.
For the full key reference, see
[Skip-Softmax](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends/trtllm.md#skip-softmax)
and
[SAGE quantization](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends/trtllm.md#sage-quantization)
in the TRTLLM attention guide.

## Text encoder tensor parallelism

The Qwen3-VL text encoder (~51.5 GB in BF16 for the retained 50 layers) is by
default fully resident on the DiT main rank. On multi-GPU no-offload runs that
rank becomes the peak-memory hotspot. The following configuration shards the
encoder across four ranks; see the model guide for the
[tensor-parallel contract](MiniMax-H3.md#text-encoder-tensor-parallelism).

```bash
vllm serve MiniMaxAI/MiniMax-H3 \
  --omni \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling
```

Valid values of `N` on a 4-GPU server are 1, 2, and 4 (1, 2, 4, 8 on 8 GPUs).
Sharding drops the DiT main rank's no-offload peak by roughly `(N-1)/N` of the
~51.5 GB encoder while the other ranks each gain `~51.5/N` GB.

## Validation results

### Validated four-GPU evidence

The four-GPU recommendation was measured on four NVIDIA B300 GPUs with one
excluded warmup followed by three requests.

| Workload                               | Configuration                               | Observed result                      |
| -------------------------------------- | ------------------------------------------- | ------------------------------------ |
| FL2VA, 209 frames, 1248x768            | no offload, U4, VPP4 tile, regional compile | 86.964 s mean HTTP client latency    |
| Two-video Ref2VA, 362 frames, 1344x768 | no offload, U4, VPP4 tile, regional compile | 784.394 s accounted model-stage mean |

These measurements describe the validated shapes rather than a general
throughput guarantee. Multi-video Ref2VA is much slower because the two
reference videos expand both the Qwen3-VL vision sequence and the packed DiT
attention sequence.

### Validated FP8 evidence

With eager DiT/text-encoder TP2 and VAE tiling, the 384x672, 107-frame,
10-step quality case measured LPIPS 0.1156 (limit 0.20), PSNR 23.6316 dB,
audio spectral cosine 0.9589 (minimum 0.80), and audio RMS ratio 0.9342. The
resident per-GPU peak was 68.52 GiB for BF16 and 53.51 GiB for FP8, a 22%
reduction.

For direct human inspection, see the external
[BF16 versus global-FP8 comparison](https://lishunyang12.github.io/vllm-omni-rankings/scripts/minimax_h3_global_fp8_vs_bf16/),
which includes matched five-second T2VA, I2VA, and Ref2VA videos. The
[comparison sources](https://github.com/lishunyang12/vllm-omni-rankings/tree/main/scripts/minimax_h3_global_fp8_vs_bf16)
also record per-task fidelity metrics and provenance without storing generated
media in this repository.

### CPU MP4 response encoding

For CUDA and ROCm deployments, non-streaming MP4 responses are encoded on the
host CPU through PyAV/libx264 after generation. The response encoder selects the
path automatically at runtime. The server-owned parallel converter accepts
supported frame shapes and dtypes with either per-channel-contiguous or strided
RGB planes, including interleaved arrays materialized by output transport.
Standalone callers without a parallel converter retain the legacy fallback for
strided planes. No CLI flag, model declaration, or user configuration is
required. Streaming fMP4 output is unchanged.

A community benchmark on 2x Xeon 8480C reported the following comparison
between the legacy and direct planar paths
([full result](https://github.com/vllm-project/vllm-omni/pull/6288#issuecomment-5337546499)):

| Metric | Legacy | Direct planar | Change |
| --- | ---: | ---: | ---: |
| Median wall time | 1.805 s | 1.394 s | -22.8% |
| Median process CPU time | 3.613 s | 3.207 s | -11.2% |
| Peak RSS | 3182 MiB | 2794 MiB | -387 MiB (-12.2%) |

Across the 1.0-8.7 s sweep, wall-time improvement was approximately 21.8-22.6%;
outputs were byte-identical and full decode passed. This is evidence for host
CPU response encoding. Actual gains depend on the CPU and runtime, and should
not be interpreted as GPU, DiT, or stage 0 speedups.

### FastH3 Dense validation

The following measurements use a separate eight-GPU Dense configuration:

Measured on 8x NVIDIA B300 with USP8, VAE patch-parallel 8, `TRTLLM_ATTN`, at
1344x768, 4.4 s, seed 1101, one warmup excluded and two runs recorded:

| Adapter | Steps | End-to-end | Diffusion engine |
| --- | ---: | ---: | ---: |
| none (base H3) | 50 | 25.8 / 26.4 s | 16.22 / 16.28 s |
| FastH3 Dense | 4 (5 sigma points) | 11.7 / 11.8 s | 2.37 / 2.36 s |

The denoising speedup is 6.9x. End-to-end is 2.2x because text encoding, VAE
decoding and muxing are a fixed cost that dominates a clip this short; longer
generations move the end-to-end figure toward the denoising one. Fusing the
adapter does not measurably change startup: weight loading took 77.3 s with it
against 85.8 s without.

### Step execution measurements

!!! warning "Co-batching does not improve H3 throughput for large simultaneous requests"
    Keep `--max-num-seqs 1` unless you specifically need scheduler-level control
    (admitting and retiring requests between denoise steps). Measured on two
    H100s (TP2, BF16, 672x384, 209 frames, 30 steps, 4 requests at concurrency
    4 with `--request-rate inf`, i.e. all four submitted at time 0; one packed
    request is 16384 rows):

    | Configuration | Wall time | Mean latency | Peak memory |
    |---------------|-----------|--------------|-------------|
    | request mode | 174.8 s | 111.5 s | 72.4 GB |
    | `--step-execution --max-num-seqs 1` | 179.0 s | 113.8 s | 72.4 GB |
    | `--step-execution --max-num-seqs 4` | 182.1 s | 175.7 s | 78.3 GB |

    A single H3 denoise step is a compute-bound dense GEMM over an already long
    packed sequence, so fusing N requests costs N times the FLOPs and buys almost
    no amortization — unlike LLM decoding, which is memory-bandwidth bound.
    Going from one request per step to four cuts the per-request denoise cost
    only from 1.323 s to 1.291 s (2.4%), which the step-mode bookkeeping then
    spends. Mean latency degrades further because co-batched requests finish
    together instead of staggered. Quantization moves the absolute numbers
    without changing this: with online `int8` the same workload runs in 153.3 s
    at 56.9 GB (request mode), and `--max-num-seqs 4` is still 5.0% slower than
    request mode.

    Two workloads outside this table are unmeasured and may behave differently:

    - **Staggered arrivals (admission latency).** A single 16384-row H3 request
      already saturates the GPUs, so submitting all requests at time 0 is the
      one arrival pattern where co-batching cannot win: it can only bunch
      completions. In request mode a new request queues behind the whole
      in-flight generation (~45 s at this size); step mode admits at the next
      denoise-step boundary (~1.3 s). Whether that shows up as a wall-clock
      benefit under Poisson arrivals is not yet measured for H3. To reproduce,
      run request mode vs `--step-execution --max-num-seqs 4` with
      `diffusion_benchmark_serving.py --request-rate 0.05` (roughly one request
      every 20 s) and report mean / p95 latency, which then includes queueing.
    - **Small requests.** The numbers above are drawn from 672x384 / 209-frame
      requests. A short clip at lower resolution packs a few thousand rows and
      may not saturate the hardware; co-batching may amortize better there.
      This is also unmeasured.

### Single 96 GB GPU, no-offload capacity check

Use the FL2VA-only partition for this capacity test. Loading the combined
service would also load the Ref2VA DiT and would test a different memory
budget. A no-offload capacity check should omit
`--diffusion-offload-config` and all legacy `--enable-*-offload` aliases. VAE
tiling changes decode placement but does not offload model weights to the CPU.

The run passes the capacity check when the server initializes, the request
finishes without CUDA OOM or Xid errors, `peak_used_mib` remains below the
card's reported `memory_total_mib`, and `ffprobe` reports H.264 video plus
32 kHz stereo AAC audio. Report the measured headroom rather than assuming
that every nominal 96 GB SKU exposes the same MiB total.

As a capacity proxy only, the same five-second case on one B300 measured a
92,946 MiB whole-device first-request peak and a 92,146 MiB worker peak. Its
encode, diffuse, decode, and client wall times were 8.664 s, 134.504 s,
6.016 s, and 151.583 s. These numbers suggest that a 96 GB card may fit, but
they are not an RTX PRO 6000 validation: kernels, allocator behavior, and
reported device capacity differ across GPUs.

### Request-scoped quality validation

The following result was measured on 4× NVIDIA H200 with SP4, text-encoder
TP4, 1344×768, 124 frames, 24 FPS, and 50 inference steps. One full
`lossless` warmup was excluded, followed by three fixed prompt/seed pairs in
balanced switch order.

| `quality` | Median inference latency | Speedup | SSIM vs `lossless` | PSNR vs `lossless` | Expected trade-off |
| --- | ---: | ---: | ---: | ---: | --- |
| `lossless` | 85.49 s | 1.00× | 1.0000 | exact | Native reference path |
| `high` | 63.36 s | 1.35× | 0.9709 | 34.98 dB | Faster with measured same-seed deviation |

> [!NOTE]
> `high` selects a fixed Cache-DiT profile, but its cache hit rate and
> resulting latency/quality trade-off may vary by hardware, topology, and
> workload. The values above apply to this deployment and are not universal
> guarantees. `lossless` remains the exact reference path.
