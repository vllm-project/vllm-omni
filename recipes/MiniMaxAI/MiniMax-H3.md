# MiniMax H3

> Joint video and audio generation with text, first/last keyframes, and
> mixed image/video/audio references

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Maintainer: Community

MiniMax H3 is a CFG-distilled joint video/audio diffusion transformer. Its
checkpoint has two task-specific DiT partitions:

- `FL2VA`: text-to-video+audio (`t2va`) and first-frame-to-video+audio
  (`fl2va`)
- `Ref2VA`: up to 9 images, 3 videos, and 3 audio references in supported
  image/video/audio combinations (`ref2va`; audio-only is rejected)

One vLLM-Omni diffusion stage can load both DiTs while instantiating the
tokenizer, processor, Qwen3-VL text encoder, video VAE, and audio VAE only
once. Requests select the DiT with `extra_params.task`.

This single-stage recipe uses the default `model_loaded.text_encoder: true` and
`model_loaded.vae_encoder: true`. Precomputed text conditioning can replace the
local Qwen call. Setting `model_loaded.text_encoder: false` while leaving
`model_loaded.vae_encoder: true` requires precomputed text conditioning;
reference media is still encoded locally.

The generated MP4 contains H.264 video and synchronized stereo audio.

## Prerequisites

The checkpoint requires Hugging Face access approval. Authenticate once;
`vllm serve` downloads the required components automatically:

```bash
hf auth login
export MODEL=MiniMaxAI/MiniMax-H3
```

By default, the pipeline downloads `FL2VA/**`, `Ref2VA/model_index.json`, and
`Ref2VA/transformer/**`. It does not download or load the diffusers-format
`transformer`, `transformer_ref`, or `vae` weights at the repository root, nor
duplicate Ref2VA copies of shared components.

Install vLLM-Omni from the checkout containing MiniMax H3 support. The
two-GPU RTX 5090/4090 profiles use cuDNN attention and do not need
FlashAttention-4. Install the optional dependency only for the four-GPU
B300/GB200 `FLASH_ATTN` profile:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

To keep FA4 available as an explicit option on Blackwell, install the
FlashAttention-4 extra:

```bash
uv pip install -e '.[fa4]'
```

On AMD ROCm, install without the `[fa4]` extra (FA4 is CUDA-only) and use
`--diffusion-attention-backend FLASH_ATTN`; see
[AMD ROCm (gfx942 / gfx950)](#amd-rocm-gfx942--gfx950).

`ffmpeg` and `ffprobe` must be available on `PATH`. They are used for
reference-video preparation and MP4 output.

## CPU MP4 response encoding

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

## Start a server

Pass the repository ID directly. The pipeline uses `FL2VA` for model discovery
and shared components, and loads the second DiT from `Ref2VA/transformer`.

### Memory and storage requirements

Treat GPU HBM, host RAM, and checkpoint storage as separate requirements. Each
H3 checkpoint partition (`FL2VA` or `Ref2VA`) contains about **134 GiB** of
BF16 safetensors (about **135 GiB** on disk). Keeping both partitions
locally therefore needs roughly **270 GiB** of model storage. A combined
service downloads both; `--task-type fl2va` or `--task-type ref2va` downloads
only the selected partition.

CPU offload and distributed layerwise offload reduce GPU residency; they do
not make the model weights disappear. With `--dlo-no-use-allgather`, each
worker retains its standard-loader rank-local weights in host memory, including
pinned CPU buffers used for H2D streaming. Use at least **200 GiB available
system RAM** before starting the two-GPU recipe; a **384 GiB host is
recommended** to leave room for the OS, CUDA/PyTorch allocations, request
inputs, and filesystem cache. Do not run the FL2VA and Ref2VA servers at the
same time on a host sized for this minimum.

The consumer-GPU profiles below are HBM budgets only. They still require the
host-RAM budget above.

### Single GPU: blockwise capacity path

Use ordinary layerwise offload when one GPU cannot keep the Qwen3-VL encoder
and DiT resident together. The component list below streams the active DiT,
the Qwen vision blocks, and the first 50 Qwen text layers. Encoder blocks stay
rank-local; the video/audio VAEs remain resident.

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
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

### Two 24/32 GB GPUs: TP2 distributed layerwise offload

For two PCIe consumer GPUs, combine TP2 with distributed layerwise offload
(DLO). The standard loader first creates the rank-local TP shard. DLO keeps
that shard in pinned host memory and streams the 30 tail DiT blocks through a
shared two-buffer window without a DP AllGather. The first 20 DiT blocks are
copied to the GPUs once per denoise stage, reused by every sampling step, and
released before VAE decode so the decoder can reuse their HBM.

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --dlo-resident-layers 20 \
  --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

Use the profile that matches the per-GPU memory capacity:

| Profile | GPUs | Starting shape | Resident DiT blocks | Attention | Execution | Status |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `rtx5090` | 2 x 32 GB | 1344x768 | 20 | cuDNN attention | eager | Target-hardware validated |
| `rtx4090` | 2 x 24 GB | 1024x576 | 12 | cuDNN attention | eager | Capacity-proxy starting point |

This topology uses all available parallel capacity: TP2 shards both the DiT
and text encoder, `--dlo-no-use-allgather` streams each rank's local TP shard
without reconstructing full blocks, and VAE patch parallelism splits tiled
decode across both GPUs. cuDNN attention is selected explicitly for the RTX
consumer path; the server stays eager to avoid an unqualified compile path.

The resident count changes placement and transfer frequency only; it does not
quantize or change the BF16/FP32 denoise math. Re-measure peak memory before
increasing it on a different request shape.

### RTX 5090 target-hardware validation

At vLLM-Omni commit `ae6577ea`, one full 50-step T2VA request completed on
2 x RTX 5090 without OOM:

| Shape    | Frames        | Client E2E | Sampled peak/GPU       | Output validation                                            |
| -------: | ------------: | ---------: | ---------------------: | -----------------------------------------------------------: |
| 1344x768 | 124 at 24 FPS | 8 min 38 s | approximately 22.6 GiB | H.264 video + 32 kHz stereo AAC; full `ffmpeg` decode passed |

This is a single end-to-end validation run, not a warmed multi-run latency
benchmark. The sampled `nvidia-smi` peak is also not a CUDA allocator
high-water mark. The environment used vLLM 0.26.0, vLLM-Omni
`0.26.1.dev14+gae6577ea`, and PyTorch 2.11.0+cu130. The
[run record](https://github.com/lishunyang12/vllm-omni-rankings/blob/dcd06d7e83cb069842535918c0169ee9f3f29ba0/scripts/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20260805000034_86_237.png)
captures the environment, output contract, elapsed time, and sampled peak.

Before the target run, both profiles were exercised on two B300 ranks as an
allocation and correctness proxy. At 1344x768, 124 frames, and 50 steps, the
20-layer profile peaked at 27,726 MiB per rank. At 1024x576, the 12-layer
profile peaked at 18,888 MiB per rank in a 5-step capacity run. The resident
and fully streamed placements produced identical decoded video-frame and audio
hashes for the same shape, step count, prompt, and seed. The B300 result does
not establish RTX 4090 PCIe latency; treat the 4090 profile as a conservative
starting point until it is measured on that GPU.

To run T2VA, FL2VA, image+audio Ref2VA, and two-video Ref2VA in order, validate
every MP4's H.264/AAC streams, and retain live server and GPU-memory logs:

```bash
RUN_ROOT=/path/to/run-root \
MODEL_ROOT=/path/to/MiniMax-H3 \
GPU_IDS=0,1 \
PROFILE=rtx5090 \
bash examples/offline_inference/minimax_h3/run_h3_2gpu_all_tasks.sh
```

The script selects 20 resident layers for `PROFILE=rtx5090` and 12 for
`PROFILE=rtx4090`; `DLO_RESIDENT_LAYERS=N` overrides either default.

### Four GPUs: throughput-oriented combined service

For a combined service on four high-memory GPUs, use:

- no CPU or layerwise offload;
- Ulysses sequence parallelism degree 4;
- native tiled VAE patch parallelism degree 4;
- regional `torch.compile` for the repeated DiT blocks;
- dense BF16 `TRTLLM_ATTN`, with Ring and TP left at 1.

Both DiTs remain resident in this no-offload configuration. If they do not fit,
use model-level CPU offload.

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
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

### Attention Backends

On supported datacenter Blackwell systems, MiniMax H3 defaults to dense BF16
`TRTLLM_ATTN`; no attention backend flag is required. To select it explicitly,
use:

```bash
--diffusion-attention-backend TRTLLM_ATTN
```

Stable measurements with the four-GPU profile above put dense `TRTLLM_ATTN`
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
attention. Both work under the pure Ulysses parallelism of the profile above
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

### Text encoder tensor parallelism

The Qwen3-VL text encoder (~51.5 GB in BF16 for the retained 50 layers) is by
default fully resident on the DiT main rank.  On multi-GPU no-offload runs that
rank becomes the peak-memory hotspot.  Add `--text-encoder-tp-size N` to shard
the encoder across the first `N` DiT ranks (the encoder is implemented with
vLLM-style tensor-parallel layers and runs with distributed collectives over
its own encoder process group):

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling
```

`N` must divide the Qwen3-VL head counts (64 attention heads / 8 KV heads), so
valid values on a 4-GPU server are 1, 2, and 4 (1, 2, 4, 8 on 8 GPUs).  The
encoder TP rank set is the first `N` DiT ranks; on 4 GPUs
`--text-encoder-tp-size 4` shards the encoder 4-way, dropping the DiT main
rank's no-offload peak by roughly `(N-1)/N` of the ~51.5 GB encoder while the
other ranks each gain `~51.5/N` GB.  The encoder output remains identical to
the reference path within bf16 rounding: every encoder rank all-reduces the
row-parallel projections, so the full `[seq, 5120]` layer-50 hidden state is
replicated on every rank.

No restart is needed: `task=fl2va` routes to `FL2VA/transformer`, while
`task=ref2va` routes to
`Ref2VA/transformer`. T2VA uses the FL2VA DiT.

### Step execution and continuous batching

H3 implements the step-wise execution contract, so the scheduler can admit and
retire requests between denoise steps instead of running one request end to
end. Add the feature gate, then raise `--max-num-seqs` to co-batch:

```bash
--step-execution --max-num-seqs 4
```

Co-batched requests are packed into a single sequence that keeps one attention
document per request, so a batch costs one DiT forward. That packing requires
`--diffusion-attention-backend FLASH_ATTN`; other backends fall back to one
forward per request. `--max-num-seqs 1` keeps the conservative single-request
step path. Cache acceleration (`--cache-backend`) is not available in step mode.

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

### Online FP8 quantization

MiniMax H3 supports online FP8 quantization of both the DiT and the Qwen3-VL
text decoder. The checkpoint remains BF16 on disk; vLLM-Omni creates FP8
weights at runtime and uses dynamic activation scaling during inference. By
default, `--quantization fp8` quantizes eligible attention and MLP linears in
the text decoder, token refiner, and main DiT blocks, as well as the DiT
condition and AdaLN projections. The Qwen vision tower, embeddings, norms,
RoPE, both VAEs, and the model's FP32 patch, timestep, and output projections
keep checkpoint precision.

To select a component, use `--diffusion-quantization-config` with
`{"transformer":{"method":"fp8"}}` for DiT-only FP8 or
`{"text_encoder":{"method":"fp8"}}` for text-decoder-only FP8. The two
entries can be combined. The shorthand below enables both components.

Add this option to an existing H3 server command:

```bash
--quantization fp8
```

#### Single 96 GB GPU, no-offload capacity check

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

Use `ignored_layers` to keep any otherwise eligible linear in BF16. H3
resolves the `transformer` component before constructing the DiT, so names do
not start with `transformer.`. Entries are exact runtime linear prefixes; a
parent name such as `blocks.0.attn` does not exclude its children.

Eligible names are the `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and
`mlp.fc2` children under `token_refiner.blocks.<0-1>` or `blocks.<0-49>`.
The other eligible names are `condition_proj`,
`blocks.<0-49>.adaln_proj.linear`, and `final_layer.adaln_proj.linear`.
For example, keep the first main block's attention projections in BF16 with:

```bash
--diffusion-quantization-config \
  '{"transformer":{"method":"fp8","ignored_layers":["blocks.0.attn.qkv_proj","blocks.0.attn.out_proj"]}}'
```

The structured option replaces `--quantization fp8`. Online FP8 can be used
with H3 layerwise offload and with either DiT DLO transfer. DiT `allgather`
uses the ordinary loader to finalize FP8 weights and scales before sharding
them across ranks. DiT `rank-local` instead retains complete loader-produced
tensors and avoids the synchronized request-wave contract. H3's TP-sharded
text encoder uses `rank-local`.

## AMD ROCm (gfx942 / gfx950)

The sections above describe the NVIDIA CUDA path. This section covers the AMD
ROCm path; use it instead of the CUDA commands when running on AMD Instinct
GPUs.

MiniMax H3 runs on AMD Instinct GPUs (gfx942 / gfx950) in BF16. Use
`--diffusion-attention-backend FLASH_ATTN`, which resolves to AITER packed varlen
attention on both architectures.

Select the device with `HIP_VISIBLE_DEVICES` (not `CUDA_VISIBLE_DEVICES`), drop the
CUDA-only `FLASHINFER_DISABLE_VERSION_CHECK`, and install without the `[fa4]` extra
(FA4 is CUDA-only). The VAE uses AITER GroupNorm on ROCm.

Install (ROCm wheel + source vLLM-Omni):

```bash
pip install "vllm==0.26.0+rocm723" \
  --extra-index-url https://wheels.vllm.ai/rocm/0.26.0/rocm723
VLLM_OMNI_TARGET_DEVICE=rocm pip install -e . --no-build-isolation
```

Prebuilt image: `vllm/vllm-omni-rocm:minimax-h3`. All tasks work out of the box:
the image bundles TorchCodec (for image+audio Ref2VA) and `ffmpeg` (for
video-reference Ref2VA).

### ROCm single GPU

Single GPU with model-level CPU offload keeps the Qwen3-VL encoder and DiT from
being co-resident:

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

HIP_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni --host 0.0.0.0 --port "${PORT}" --trust-remote-code \
  --num-gpus 1 --enable-cpu-offload \
  --diffusion-attention-backend FLASH_ATTN
```

This validated ROCm capacity recipe intentionally retains the compatibility
full-topology alias because MiniMax-H3 also stages its VAEs on that path. The
compact API in this release selects only `dit` and `text_encoder`, so replacing
the flag would change residency rather than perform a mechanical migration.
No removal deadline is assigned until the compact API offers equivalent
component coverage.

### ROCm four GPUs

The best-practice CUDA four-GPU configuration works on ROCm with the changes above.
It mirrors the CUDA command including Ulysses sequence parallelism, VAE patch
parallelism, and text-encoder tensor parallelism, with no CPU offload when the model
shards across the GPUs:

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

HIP_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --usp 4 \
  --ring 1 \
  --text-encoder-tp-size 4 \
  --vae-patch-parallel-size 4 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

As on CUDA, H3 is CFG-distilled, so keep `--cfg-parallel-size` at 1, and the H3 VAE
supports only its native `tile` mode. `--text-encoder-tp-size` is validated on
gfx942; on gfx950 it was not exercised.

### Validated ROCm evidence

vLLM-Omni with MiniMax H3 support, BF16. gfx942 rows measured with the
`vllm/vllm-omni-rocm:minimax-h3` image; gfx950 rows measured with the
`0.26.0+rocm723` wheel (HIP 7.2).

| Workload | Configuration | Observed result |
| ---------- | --------------- | ----------------- |
| T2VA, 1344x768, 209 frames, 50 steps | 4x gfx942 (MI300X), FLASH_ATTN, USP4, text-enc TP4, VAE PP4 tile | encode 0.09 s, denoise 244.04 s, decode 4.15 s, 267.42 s client E2E; H.264 24 FPS + 32 kHz stereo AAC |
| FL2VA, 1344x768, 209 frames, 50 steps | 4x gfx942 (MI300X), FLASH_ATTN, USP4, text-enc TP4, VAE PP4 tile | encode 13.98 s, denoise 257.58 s, decode 4.11 s, 287.07 s client E2E; H.264 24 FPS + 32 kHz stereo AAC |
| T2VA, 832x480, ~4 s, 40 steps | 1x gfx950 (MI350), FLASH_ATTN, CPU offload | valid MP4 (H.264 + synced audio); ~0.73 s/denoise-step (~1.37 it/s), ~55 s client E2E incl. warmup |

gfx942 figures are the mean of three requests after one excluded warmup
(external evidence: vllm-project/recipes#732). gfx950 figures are
functional-correctness validations, not tuned throughput; the first request
includes lazy regional compilation. MI325X (gfx942) and other MI355X SKUs are not
listed until their own evidence is added.

## HTTP API examples

The following requests use the synchronous endpoint so the returned body can
be saved directly as an MP4. The asynchronous `POST /v1/videos` endpoint can
also be used when job polling is preferred.

All four tasks use 24 FPS, 50 sigma points, seed values from the validated
workloads, and the checkpoint-reference video/audio flow shifts of 12 and 3.
Decimal durations are passed through `extra_params`.

Set the endpoint once:

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
```

### 1. T2VA: text to video and audio

Run this request against the combined service:

```bash
curl -sS -X POST "${API_URL}" \
  -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

### 2. FL2VA: first frame to video and audio

Run this request against the combined service. When width and height are
omitted, H3 preserves the first-frame aspect ratio and uses a 768-pixel short
edge.

```bash
export FIRST_FRAME=/path/to/fl2va_first_frame.png

curl -sS -X POST "${API_URL}" \
  -F 'prompt=A man stands beside a yellow car at night. The car drives away; he follows it with his eyes and begins singing sadly, with synchronized voice and city ambience.' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=2101' \
  -F 'extra_params={"task":"fl2va","duration":8.7,"audio_flow_shift":3.0}' \
  -F "input_reference=@${FIRST_FRAME};type=image/png" \
  -o fl2va.mp4
```

Use the same `FL2VA` partition for the official tail-keyframe forms. A single
image with `frame_indices=[-1]` conditions the last frame; two ordered images
with `frame_indices=[0,-1]` condition the first and last frames:

```bash
export LAST_FRAME=/path/to/fl2va_last_frame.png
export FIRST_FRAME=/path/to/fl2va_first_frame.png

curl -sS -X POST "${API_URL}" \
  -F 'prompt=The subject moves naturally from the first image to the last image.' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=2102' \
  -F 'extra_params={"task":"fl2va","duration":8.7,"frame_indices":[0,-1],"audio_flow_shift":3.0}' \
  -F "input_references=@${FIRST_FRAME};type=image/png" \
  -F "input_references=@${LAST_FRAME};type=image/png" \
  -o fl2va_first_last.mp4
```

### 3. Ref2VA: image-only, image/audio, or mixed references

Run these requests against the combined service or a Ref2VA-only service.
Image-only Ref2VA omits `audio_reference`; adding one or more audio references
is optional. The typed fields accept one object or an ordered JSON list.
`audio_reference` accepts an HTTP(S) URL or a `data:` URL. In one terminal,
expose the local reference assets to the serving host:

```bash
python -m http.server 8092 \
  --bind 127.0.0.1 \
  --directory /path/to/reference_assets
```

Then submit an image-only request from another terminal:

```bash
export REF_IMAGE=/path/to/reference_assets/ref2va_image.png

curl -sS -X POST "${API_URL}" \
  -F 'prompt=A white cat sits on a beige couch and slowly looks toward the camera.' \
  -F 'aspect_ratio=adaptive' \
  -F 'short_edge=768' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3100' \
  -F 'extra_params={"task":"ref2va","duration":8.0,"audio_flow_shift":3.0}' \
  -F "input_reference=@${REF_IMAGE};type=image/png" \
  -o ref2va_image_only.mp4
```

An image-plus-audio request is:

```bash
export REF_IMAGE=/path/to/reference_assets/ref2va_image.png
export AUDIO_URL=http://127.0.0.1:8092/ref2va_audio.mp3

curl -sS -X POST "${API_URL}" \
  -F 'prompt=A white cat with black mustache and eyebrow markings sits on a beige couch, lip-syncing precisely to the complete reference audio before shifting from confusion to deadpan speechlessness.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3101' \
  -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
  -F "input_reference=@${REF_IMAGE};type=image/png" \
  -F "audio_reference={\"audio_url\":\"${AUDIO_URL}\"}" \
  -o ref2va_image_audio.mp4
```

The requested duration should cover the complete audio. If `duration` is
shorter, the reference soundtrack is truncated to the generated clip.

### 4. Ref2VA: video, separate audio, and mixed references

Run this request against the combined service. Repeat the
`input_references` multipart field once per source video. H3 consumes the
videos in form order and preserves their original soundtracks during
conditioning.

```bash
export SUBJECT_VIDEO=/path/to/green_screen_subject.mp4
export BACKGROUND_VIDEO=/path/to/fairytale_background.mov

curl -sS -X POST "${API_URL}" \
  -F 'prompt=Remove the green screen background of Video 1 and replace it with the fairytale environment from Video 2. Match the background motion to the character actions and relight the character to fit the scene.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3101' \
  -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
  -F "input_references=@${SUBJECT_VIDEO};type=video/mp4" \
  -F "input_references=@${BACKGROUND_VIDEO};type=video/quicktime" \
  -o ref2va_video_video.mp4
```

The server stores uploaded references only for the lifetime of the request and
deletes temporary files after generation. A video may use its embedded
soundtrack, a separate `audio_reference`, or both. To send a mixed multipart
request, repeat `input_references` for each image, video, or audio file; the
server classifies them by MIME type and preserves the per-type order.

Reference videos must be MP4/MOV with H.264/H.265 video, optional AAC/MP3
audio, 2–15 seconds each, and at most 15 seconds combined. They may still be
longer than the generated clip. Use `start_time_seconds` to select a
synchronized segment; for multiple typed video references, pass one value per
video in `extra_params.start_time_seconds`.

Reference images accept JPG/JPEG, PNG, WEBP, HEIC, or HEIF up to 30 MiB. Standalone
audio references accept WAV or MP3 up to 15 MiB, with 2–15 seconds per file and
at most 15 seconds combined.

## Fun ControlNet Union (CTRL-01)

The original [Alibaba PAI Fun ControlNet Union checkpoint](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union)
adds one control branch to the existing FL2VA transformer. It accepts prepared
Canny, Depth, HED, MLSD or Pose videos and mask-based video inpainting. The
control checkpoint contains only branch weights; keep the base FL2VA components
available. ComfyUI-repacked, pruned and quantized control checkpoints are not
supported by this loading path.

Set local checkpoint paths and start the control-enabled server:

```bash
export MODEL=/path/to/MiniMax-H3/FL2VA
export CONTROL_MODEL=/path/to/MiniMax-H3-Fun-Controlnet-Union.safetensors
CUDA_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve "$MODEL" --omni --trust-remote-code --task-type fl2va \
  --served-model-name MiniMaxAI/MiniMax-H3 --host 127.0.0.1 --port 8092 \
  --controlnet-model-path "$CONTROL_MODEL" \
  --num-gpus 2 --tensor-parallel-size 2 --text-encoder-tp-size 2 \
  --usp 1 --ring 1 --enforce-eager --diffusion-attention-backend TORCH_SDPA \
  --vae-patch-parallel-size 2 --vae-parallel-mode tile --vae-use-tiling
```

The initial control path uses resident BF16 weights and tensor parallelism.
Reference/keyframe conditioning, sequence parallelism, cache acceleration and
quantized control execution require separate support; do not combine them with
this configuration. Control/Turbo combinations need their own validation and
are not established by ordinary H3 Turbo results.

Upload a prepared Canny video through the existing control API:

```bash
curl --fail-with-body http://localhost:8092/v1/videos/sync \
  -F 'model=MiniMaxAI/MiniMax-H3' \
  -F 'prompt=A forest stream with flowing water and birdsong.' \
  -F 'control_type=canny' \
  -F 'control_reference=@canny.mkv;type=video/x-matroska' \
  -F 'extra_params={"task":"t2va","canny":{"control_context_scale":1.0},"audio_flow_shift":3.0}' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' \
  -F 'num_frames=124' -F 'fps=24' -F 'num_inference_steps=40' \
  -F 'guidance_scale=1.0' -F 'flow_shift=12.0' -F 'seed=43' \
  -o control-output.mp4
```

Use `depth`, `hed`, `mlsd` or `pose` for the corresponding prepared hint, and
use that same key in `extra_params`. Selecting a type does not run a
preprocessor. Strength is the model's `control_context_scale`: default 1.0,
finite and nonnegative; 0 disables the control branch.

For inpainting, source and mask have explicit roles:

```bash
curl --fail-with-body http://localhost:8092/v1/videos/sync \
  -F 'model=MiniMaxAI/MiniMax-H3' \
  -F 'prompt=A small wooden bridge crosses the forest stream.' \
  -F 'control_type=inpaint' \
  -F 'source_reference=@source.mp4;type=video/mp4' \
  -F 'mask_reference=@mask.png;type=image/png' \
  -F 'extra_params={"task":"t2va","inpaint":{"control_context_scale":1.0},"audio_flow_shift":3.0}' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' \
  -F 'num_frames=124' -F 'fps=24' -F 'num_inference_steps=40' \
  -F 'guidance_scale=1.0' -F 'flow_shift=12.0' -F 'seed=43' \
  -o inpaint-output.mp4
```

White/1 in the mask means regenerate; black/0 supplies preserved source
conditioning. This is learned conditioning, not a promise of pixel-exact
compositing outside the mask. A static image mask repeats over the target
frames; a temporal mask uses a video. Prefer lossless mask encodings. Keep the
source, control and mask aligned to the same canvas and timeline. The model
fits conditioning to the requested frame count by truncating longer inputs or
holding the final frame of shorter inputs.

A source requires a mask. A mask without a source is supported, using zeroed
source pixels; it does not preserve an unavailable original video. A mask can
also accompany a prepared control video. Do not send the inpainting source as
`video_reference`: that field has Ref2VA semantics.

Both `/v1/videos` and `/v1/videos/sync` share these inputs. Async requests retain
the existing job polling/content interface. Unsupported controls and invalid
combinations fail instead of silently falling back to ordinary generation.
See the [video API](../../docs/serving/videos_api.md) for upload rules and errors.

### Validation status

Local integration checks used two H20-3e GPUs, driver 550.127.08, PyTorch
2.13.0+cu129, vLLM 0.29.0+cu129, Diffusers 0.40.0 and Transformers 5.14.1.
The configuration above used resident weights and TORCH_SDPA. Canny and
inpainting also completed with the native
`minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` adapter: 5 sigma
positions / 4 denoiser forwards, video/audio shifts 6/3, guidance 1, adapter
scale 1, seed 1101, 1344x768, 124 frames and 24 FPS.

To reproduce that Turbo comparison, add `--lora-backend peft --lora-path "$TURBO_LORA"`
at startup and activate the same artifact in the request, following the
[LoRA contract](#lora). Change the request to `num_inference_steps=5`,
`flow_shift=6`, and add:

```bash
-F "lora={\"name\":\"h3-turbo-v1.0\",\"path\":\"${TURBO_LORA}\",\"scale\":1.0}"
```

The Canny check used an FFV1 edge video derived from a generated forest clip;
the inpainting check used that source clip and a static PNG mask. Canny
followed the source layout; inpainting introduced a bridge in the masked
region while retaining the surrounding scene. Both output audio and video
fully decoded, but the Turbo control samples had much quieter audio than
the same-seed control-disabled sample. This remained after matching the
reference timestep precision and correcting container timestamp rounding.
A Base 40-step Canny comparison restored the audio signal level, but a
final-code Base 40-step inpainting sample was also nearly silent. Base
inpainting therefore is not established as a workaround. Audio quality
validation remains incomplete for these combinations; an independent
reference-runtime comparison is needed before attributing the limitation
to the model or the integration. These checks establish those two request paths,
not an exhaustive quality evaluation of all hint types or Turbo variants. The
40-step examples above follow the original control recipe. This does not
claim exhaustive coverage of all Base or Turbo conditioning combinations.

Four additional real-model API smokes used synthetic analytic Depth, HED,
MLSD and Pose hints with the same Turbo configuration above (4 forwards).
All returned HTTP 200 with 1344x768 H264 video, 124 frames at 24 FPS, and
32 kHz stereo AAC; full decoding passed. Their audio RMS values were:

| Hint | Audio RMS |
| --- | ---: |
| Depth | 0.02181784 |
| HED | 0.00042317 |
| MLSD | 0.00017059 |
| Pose | 0.00990357 |

These hints were analytic geometry, edges, segments and skeletons, without
learned preprocessing. They establish API/inference smoke coverage, not
real-video fidelity. Prompts and inputs differed from the forest checks;
stronger Depth/Pose signal levels do not isolate an effect of the mode name.
HED/MLSD audio remained weak. Listening and dialogue quality were not verified,
and these results do not resolve the earlier audio findings.

### ComfyUI client boundary

CTRL-01 provides serving; the remote conditioning node and full workflow belong
to CTRL-02 and WF-06. The client integration should extend the existing
Generate Video node with the explicit multipart roles above. Pose preprocessing belongs in the client workflow:

```text
Load Video -> Get Video Components -> SDPose Keypoint Extractor
           -> SDPose Draw Keypoints -> Create Video (24 FPS) -> control video
```

Use the detector and model loader required by the pose extractor. Already
prepared pose videos can skip preprocessing. The server does not load SDPose
or a person detector merely because `control_type=pose` was selected.

## Official input matrix and limits

| Task | Supported references | Limits |
| ------ | ---------------------- | -------- |
| T2VA | text only | prompt must be non-empty |
| FL2VA | first image, last image, or ordered first+last images | at most 2 images; `frame_indices` is `[0]`, `[-1]`, or `[0,-1]` |
| Ref2VA | image-only, image+image, image+video, video+audio, and mixed image/video/audio | images ≤9, videos ≤3, audios ≤3, total references ≤12; audio requires a visual reference |

The default H3 output contract is 4–15 seconds at 24 FPS, stereo 32 kHz audio, and a
32-pixel canvas multiple. T2VA requires one named output ratio from `21:9`,
`16:9`, `4:3`, `1:1`, `3:4`, or `9:16`. FL2VA always follows the first input
image's ratio and ignores a generic `aspect_ratio` override. Ref2VA defaults to
`16:9`; `adaptive` and SGLang's `auto` spelling are accepted aliases for that
default. `short_edge` controls the 768-pixel canvas and must be `768`.
`num_outputs_per_prompt` accepts 1–10 and derives each output seed as
`seed + output_index`. The asynchronous endpoint returns all
outputs; the synchronous raw-MP4 endpoint returns the first output when more
than one is requested.

## Long video and driving audio

Set `extra_params.long_video=true` to opt into durations above 15 seconds.
Ref2VA long-video requests now default to latent continuation with global
temporal RoPE positions (A+B), described below. Set `long_video_mode=full`
explicitly to sample the entire target latent at once for comparison. Both modes
retain the native `17n+5` video-frame grid, `5n+2` video-latent grid, and 40 Hz
audio latents. A request for 75 seconds becomes 1,807 frames (75.292 seconds at
24 FPS). In full mode, attention cost and activation memory grow with the full
sequence length. Reference-video and reference-audio limits remain unchanged.
Other task types retain full-mode behavior. Requests are limited to 30 seconds
in full mode and 300 seconds in continuation mode; without `long_video=true`,
the limit remains 15 seconds. The limits cover duration aliases and explicit
`num_frames`, and externally encoded requests are checked again before diffusion.
Native frame-grid alignment can round the output slightly above the requested
limit. These are request sanity bounds, not a guarantee that a given resolution
fits GPU memory. Full mode still allocates the whole sequence, and continuation
retains cumulative latents and reference inputs.

Use `audio_mode=lock_source` with one `audio_reference` to drive generation
with a soundtrack, rather than treating it as a short reference. The encoder
places that waveform in the target audio latent, crops or zero-pads its two
channels independently to the output duration, and keeps those rows clean
(timestep 1) throughout sampling. This mode does not insert an `<Audio 1>`
reference tag. It supports request and step execution; the default `native`
mode continues generating audio normally. Returned audio is the audio VAE's
reconstruction, not a byte-for-byte copy of the input recording.

Against a Ref2VA server, using the same asset-server setup as above:

```bash
curl --fail-with-body -sS -X POST "${API_URL}" \
  -F 'prompt=<prompt.txt' \
  -F 'width=960' -F 'height=544' -F 'fps=24' \
  -F 'num_inference_steps=50' -F 'flow_shift=12' -F 'seed=19960422' \
  -F 'extra_params={"task":"ref2va","duration":75,"long_video":true,"audio_mode":"lock_source","audio_flow_shift":3,"preencode_mp4":true,"preencode_batch_frames":17}' \
  -F "input_reference=@${REF_IMAGE};type=image/png" \
  -F "audio_reference={\"audio_url\":\"${AUDIO_URL}\"}" \
  -o long-video.mp4
```

`preencode_mp4` decodes and muxes temporal chunks on the worker, avoiding a
full decoded long video in the API process. It does not reduce the denoiser's
sequence length. For Turbo adapters, use the matching adapter's step count
and flow shift from the Turbo table below.

The [RunningHub workflow](https://www.runninghub.cn/post/2100530217365364737/)
uses T8's `MiniMaxH3AudioConditioningT8`, a full-length latent, and
`MiniMaxH3DualClockSamplerT8`; its `remix_source` strength is zero, equivalent
to source locking. This differs from rolling-overlap continuation nodes.
See [T8 conditioning](https://github.com/T8mars/comfyui-minimax-h3-audio-T8)
and [ComfyUI H3 masking](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py).
The workflow's custom Dasiwa weights, Sol attention, and T8-specific LoRA are
not supplied by this feature.

### Latent-tail continuation with global temporal positions (A+B)

`long_video=true` selects this mode by default for Ref2VA. It can also be selected
explicitly with `long_video_mode=continuation`. This experimental mode supports
Ref2VA request execution with uncached denoising (`quality=lossless`); step
execution is rejected. Use `long_video_mode=full` explicitly for whole-target
sampling, including requests up to 30 seconds that require step execution.
Continuation currently rejects latent-mask editing (`video_noise_mask` or
`audio_noise_mask` with retained source regions). Full mode supports editing;
`audio_mode=lock_source` cannot be combined with audio latent-mask editing.

```json
{
  "task": "ref2va",
  "duration": 75,
  "long_video": true,
  "long_video_mode": "continuation",
  "continuation_window_frames": 277,
  "continuation_overlap_frames": 22,
  "audio_mode": "lock_source",
  "audio_flow_shift": 3,
  "preencode_mp4": true,
  "preencode_batch_frames": 17
}
```

These defaults sample seven windows for a 1,807-frame output. Each continuation
uses the preceding AV latent tail as fixed condition rows at the new target's
time origin, samples a fresh unmasked target, discards its hidden overlap, and
appends only the new suffix. The reference image and full prompt remain present
in every window. Both window and overlap use the `17n+5` frame grid; the window
must be 107..345 frames and larger than the overlap. A final window may be shorter.
Audio boundaries are rounded on the cumulative 24 FPS / 40 Hz timeline to avoid
drift; a locked driving track is sliced separately for each stereo channel.

Before RoPE evaluation, every window adds `start_frame * 40 / 24` to the temporal
position IDs of its target audio/video, reference audio/video, and latent-tail
guides. The start includes the overlap (not just the newly appended frames).
Text, static image references, spatial coordinates, and padding remain unchanged.
Offsets retain fractional precision independently of rounded audio slice indices.
For the default windows starting at frames 0, 255, 510, the added positions are
0, 425, 850. This follows the media-only global-offset convention in
[LongMedia temporal positioning](https://github.com/vizart-vj/ComfyUI-MiniMax-H3-LongMedia/blob/main/temporal_positioning.py).
The shifted layout is constructed once per window and then used by all denoising
steps and sequence-parallel ranks; no RoPE-frequency scaling is applied.

For a narrative with different scenes, pass `continuation_prompts` as a JSON
list containing exactly one non-empty prompt string per planned window (seven
strings for the default 75-second request). These prompts are independently
encoded with the same reference assets and selected in window order. Shot times
inside each prompt describe that local window, including its hidden overlap.
This option requires a local text encoder; external-encoder stage execution is
not supported. Media positions use one shared origin after the longest encoded
text prefix, so different prompt lengths do not shift the global AV clock.
Without this list, the request's single prompt is reused as before.

The cumulative latent is decoded once after all windows. Denoising memory is
bounded by the window, while cumulative latent storage still grows with duration.
This follows the [ComfyUI latent-tail continuation algorithm](https://github.com/ttulttul/ComfyUI-Minimax-H3-Continuation).
Global positions can still exceed the model's trained temporal range. This
mode does not guarantee seamless cuts or adherence to absolute shot timestamps in
a repeated prompt. For comparison, keep the model, seed, prompt, source media,
steps, shifts, and output size fixed, changing only `long_video_mode`.

For a reproducible approximately 20-second comparison, use a request-mode
Ref2VA server with caching disabled and the same `prompt.txt` and `REF_IMAGE`
for both requests. Use a single prompt without `continuation_prompts` so that
both modes receive identical text. The commands below generate native audio:

```bash
for mode in full continuation; do
  curl --fail-with-body -sS --max-time 14400 "${API_URL}" \
    -F 'prompt=<prompt.txt' \
    -F "input_reference=@${REF_IMAGE};type=image/png" \
    -F 'width=960' -F 'height=544' -F 'fps=24' \
    -F 'num_inference_steps=50' -F 'flow_shift=12' -F 'seed=19960422' \
    -F 'quality=lossless' \
    -F "extra_params={\"task\":\"ref2va\",\"duration\":20,\"long_video\":true,\"long_video_mode\":\"${mode}\",\"audio_flow_shift\":3,\"preencode_mp4\":true}" \
    -o "h3-20s-${mode}.mp4" || break
done
```

Both outputs should contain 481 frames (20.042 seconds at 24 FPS).
Continuation uses windows `[0,277)` and `[255,481)`; inspect frames 276/277
(about 11.54 seconds) for visual and audio discontinuities. Compare the complete
clips for motion, identity, prompt adherence, and sound, not just frame similarity:
the trajectories differ by design. Report the tested commit, checkpoint, hardware,
settings, and both videos with any comparison; the command alone is not quality
evidence. A matching seed does not imply matching noise over differently sized
full and windowed latents.

## Request-scoped quality

Add one of these fields to any HTTP request above. No Cache-DiT startup option
is required; H3 installs its conservative profile when a `high` request
arrives and removes it for a `lossless` request.

```bash
-F 'quality=lossless'  # Native reference path for this request.
-F 'quality=high'      # H3's conservative Cache-DiT profile.
```

Omitting `quality` preserves the startup default: it uses the reference path
normally, or the server-configured profile when the server was started with
`--cache-backend cache_dit`.

Turbo is independent of this quality switch: `quality` selects a Cache-DiT
policy, while Turbo changes the active LoRA weights and sampling schedule. See
[LoRA](#lora) below.

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

## LoRA

### Turbo LoRA

The eight Diffusers-layout LightX2V Turbo artifacts are supported. The
filename records the contract, so the server reads the step count, task family
and flow shift from it and validates each request against the artifact that is
loaded. It does not rewrite request or deploy-config sampling values: the
request must carry that artifact's own settings, listed here, or it is
rejected.

| Artifact | Task | Forwards | `num_inference_steps` | `flow_shift` | declared `alpha` |
| --- | --- | ---: | ---: | ---: | ---: |
| `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | T2VA / FL2VA | 4 | 5 | 12 | none -> 8 |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` | T2VA / FL2VA | 4 | 5 | 6 | 128 |
| `minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors` | T2VA / FL2VA | 4 | 5 | 6 | 128 |
| `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` | T2VA / FL2VA | 4 | 5 | 6 | 8 |
| `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` | T2VA / FL2VA | 8 | 9 | 12 | 8 |
| `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` | T2VA / FL2VA | 8 | 9 | 6 | 8 |
| `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` | Ref2VA | 4 | 5 | 12 | 8 |
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` | Ref2VA | 8 | 9 | 6 | 8 |

`audio_flow_shift` is `3.0` across the family. Each row is the complete
published filename; use it verbatim as `TURBO_FILE` below.

Alpha needs no manual compensation: the server reads it from the artifact's
metadata, falling back to 8 with a warning for
`minimax_h3_fl2v_turbo_4step_v0.1`, the one artifact that declares none. The
request-level `scale` is a further multiplier on top of it.

> [!NOTE]
> Every artifact is rank 128 and the delta is applied at `scale * alpha /
> rank`, so the `alpha=8` rows drive at 1/16 the strength of the `alpha=128`
> rows. 8 is the default of LightX2V's reference script, which never reads the
> metadata; its documented `v0.1` command does not override that default.

Every artifact except `minimax_h3_fl2v_turbo_4step_v0.1` also ships a
`_comfyui_` export of the same weights. Those fuse Q/K/V into one projection
and are **not** supported; downloading one is refused by name. Take the
Diffusers file. The filename is the contract, so do not rename an artifact
either -- a renamed file is rejected rather than served on a guess.

FL2VA artifacts serve `t2va` and `fl2va` on any FL2VA or combined server.
Ref2VA artifacts require
`--task-type ref2va`: a combined server serves `ref2va` from a second DiT that
the adapter cannot bind to, so loading one there is refused rather than silently
running an undistilled model on the few-step schedule.

Download the artifact you want:

```bash
export TURBO_DIR=/path/to/minimax-h3-turbo
export TURBO_FILE=minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors
hf download lightx2v/Minimax-h3-Turbo "${TURBO_FILE}" --local-dir "${TURBO_DIR}"
export TURBO_LORA="${TURBO_DIR}/${TURBO_FILE}"
```

`--lora-path` accepts one artifact, or a directory holding exactly one.

> [!IMPORTANT]
> This changes earlier behaviour. `--lora-path /path/to/minimax-h3-turbo`
> pointing at a full clone of the Turbo repository used to select the v1.0 768p
> file implicitly; a directory holding several recognized artifacts is now
> rejected as ambiguous. Name the artifact instead:
> `--lora-path /path/to/minimax-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors`.

Start from a non-offloaded or DLO FL2VA server command and add
`--task-type fl2va --lora-backend peft --lora-path "${TURBO_LORA}"`.
`--lora-path` preloads the adapter; each request still activates it and must
carry that artifact's sampling settings:

```bash
-F 'num_inference_steps=5' \
-F 'flow_shift=6' \
-F 'extra_params={"task":"t2va","duration":4.4,"audio_flow_shift":3.0}' \
-F "lora={\"name\":\"h3-turbo-v1.0\",\"path\":\"${TURBO_LORA}\",\"scale\":1.0}"
```

Switching to another FL2VA artifact means repointing `TURBO_FILE`, which moves
both `--lora-path` and the request's `lora.path`, and carrying that row's
`num_inference_steps` and `flow_shift`: `9` and `6` for `8step_v1.0_768p`, `9`
and `12` for the 544p `8step_v1.0`. A request that does not match the loaded
artifact is rejected, so a mismatch cannot silently degrade output.

The two `ref2v` rows are not served by this FL2VA command. Start a
`--task-type ref2va` server, take a request from
[Ref2VA](#3-ref2va-image-only-imageaudio-or-mixed-references) and override
`num_inference_steps` and `flow_shift` with that row's values; those examples
already send `audio_flow_shift=3.0`.

For FL2VA, change `task` and add `input_reference` as shown above. This
integration is dynamic-only and does not support prefusion or LoRA composition. DLO is
supported by keeping the request-switchable LoRA A/B buffers resident on the
accelerator while DLO streams only the base blocks; budget for this additional
fixed HBM usage. Model-level and standard layerwise offload remain unsupported.
The requested sigma points always number one more than the artifact's denoiser
evaluations.

### FlashGen native LoRA

The FlashGen 4-step T2VA artifact uses the native MiniMax-H3 module layout and
declares its distilled sigma schedule in safetensors metadata. It is published on
[ModelScope](https://modelscope.cn/models/FlashGen/Minimax-H3-4step-lora-flashgen):

```text
FlashGen/Minimax-H3-4step-lora-flashgen/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
```

Download only that file:

```bash
python -m pip install modelscope
export FLASHGEN_DIR=/path/to/minimax-h3-flashgen-lora
export FLASHGEN_FILE=minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
modelscope download FlashGen/Minimax-H3-4step-lora-flashgen \
  --local_dir "${FLASHGEN_DIR}" \
  --include "${FLASHGEN_FILE}"
export FLASHGEN_LORA="${FLASHGEN_DIR}/${FLASHGEN_FILE}"
```

Start from a non-offloaded or DLO FL2VA server command and add
`--task-type fl2va --lora-backend peft --lora-path "${FLASHGEN_LORA}"`.
Each request must use T2VA and the distilled interval-count contract:

```bash
-F 'num_inference_steps=4' \
-F 'extra_params={"task":"t2va","duration":5.2}' \
-F "lora={\"name\":\"h3-flashgen-v1.0\",\"path\":\"${FLASHGEN_LORA}\",\"scale\":1.0}"
```

This path rejects Ref2VA and checkpoints that already pin `base_schedule` in
`model_index.json`. The adapter metadata carries
`base_schedule=1.0,0.7,0.4,0.15,0.0`, so `num_inference_steps=4` means four
denoiser evaluations, not five sigma points. Request-mode generation may omit
the field and take the count from the adapter schedule; `--step-execution`
requires it explicitly, because the step scheduler reads the total step count
off the request at admission, before the adapter schedule is known.

DLO is supported in request-mode generation on the same terms as the Turbo
adapter: the request-switchable LoRA A/B buffers stay resident on the
accelerator while DLO streams only the base blocks, so budget for that
additional fixed HBM usage. The native artifact is rank 64 over 259 target
modules, and its packed `qkv_proj` and `fc1` layers reuse the full-input A
tensor per slice while B carries slice-local output rows, so the resident
footprint exceeds the on-disk payload; measure it for your parallel layout
rather than assuming the checkpoint size. Pure Ulysses replicates the adapter
on every rank, while DiT tensor parallelism shards the B buffers. Model-level
and standard layerwise offload remain unsupported, and `--step-execution`
cannot be combined with `--enable-distributed-layerwise-offload`.

To validate a deployment, post the same fixed-seed T2VA request twice with the
adapter and twice without it, then compare the four output digests. The adapter
is bound and deterministic when each pair matches internally and the two pairs
differ from each other.

### FastH3 adapter

[FastH3](https://haoailab.com/blogs/fasth3-preview/) is FastVideo's four-step
DMD2 student of H3-Base. It reuses H3's text encoder, VAEs, tokenizers, and
schedulers unchanged, replacing 49 denoiser evaluations with four. It is fused
into the checkpoint at load time rather than switched per request, because it
carries full-rank deltas that no LoRA layer can express.

The bundle publishes four variants, so download one and point `--lora-path` at
it; the repository root is refused rather than guessed at:

```bash
export FASTH3_DIR=/path/to/fasth3
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors --local-dir "${FASTH3_DIR}"
export FASTH3_LORA="${FASTH3_DIR}/dense-datafree/adapter_model.safetensors"
```

Add `--task-type fl2va --lora-path "${FASTH3_LORA}"` to a non-offloaded server
command. T2VA is served by the FL2VA partition, so `--task-type fl2va` is
correct even though FastH3 preview v1 distills T2VA only. Because the adapter is
fused, `--lora-backend` does not apply and a request carrying a `lora=` field is
rejected rather than served without the adapter it asked for.

```bash
-F 'num_inference_steps=4' \
-F 'extra_params={"task":"t2va","duration":4.4}'
```

Requests must ask for `num_inference_steps=4` and `task=t2va`: the release's five
sigma points bound four denoiser evaluations, and that count is what the step
scheduler admits a request on. The server denoises on the release's own ladder,
keeping H3's per-modality shifts at the checkpoint values, so a request that
overrides `flow_shift` or `audio_flow_shift` is rejected - it would sample the
student at noise levels it was never distilled at.

Only a release that identifies itself as FastH3 is fused; any other
`fastvideo-lora-v2` adapter stays on the dynamic LoRA route. A claimed artifact
is then held to its own metadata: one that misdeclares its tensor counts or
leaves a transformer block unedited is refused at startup instead of serving
mostly base H3 weights on a four-step schedule. Offload is refused for the same
reason - `--enable-cpu-offload`, `--enable-layerwise-offload` and
`--enable-distributed-layerwise-offload` all bypass the fusion, so they fail fast.

The VSA variants are supported through FastVideo's external kernel. Install a
`fastvideo-kernel` build that provides the `fastvideo_kernel` Python module,
then add the following flags to the same command:

```bash
--diffusion-attention-backend FASTVIDEO_VSA \
--fastvideo-vsa-topk 64
```

FastH3 VSA applies its learned `.set_weight` compression gates to the complete
packed `[text | cond | audio | video]` document using the official H3 geometry:
text/condition/audio prefix tiles never cross segment boundaries, and target
video rows use `(4, 4, 4)` 3-D tiles (64 tokens). Prefix queries remain dense;
video queries select all prefix tiles plus the configured top-k video tiles.
Pure Ulysses sequence parallelism is supported: the learned gate follows the
same sequence-to-head all-to-all as Q/K/V before VSA runs. Ring and all-gather
SP remain unsupported because they do not present a complete packed sequence
to each block-sparse kernel rank.

The Dense / Data-Free variant does not require `fastvideo-kernel` and should be
served with a dense attention backend.

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

### FastH3 8-Step V2 full checkpoint

[FastH3 8-Step V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2)
ships a full Diffusers-format transformer, including learned VSA compression
gates. Serve the Hugging Face model ID or a downloaded snapshot directly:

```bash
export FASTH3_V2_MODEL=FastVideo/FastVideo-FastH3-8-Step-V2
```

Use a Hugging Face account with access to the release, and pin `--revision`
when reproducing a result. The existing component loader reads its safetensors
shards; the H3 weight loader maps names and loads Q/K/V and MLP projections into
native parameters in memory. No conversion command or second transformer
checkpoint is needed.

The transformer and text components come from the V2 release. To retain Omni's
native tiled and parallel VAE execution, the loader fetches only the video/audio
VAEs from the `MiniMaxAI/MiniMax-H3` revision pinned in V2's `provenance.json`.
Those components use the normal Hugging Face cache. No separate base-model
argument or full base-transformer download is required. Download time is a
preparation cost, separate from warmup and request latency.

Install the optional `fastvideo-kernel` build required by the
`FASTVIDEO_VSA` backend, including its H3 block-map entry point. A four-GPU
configuration is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve "${FASTH3_V2_MODEL}" --omni \
  --host 127.0.0.1 --port 8095 --trust-remote-code --task-type fl2va \
  --served-model-name FastH3-V2 \
  --num-gpus 4 --usp 4 --ring 1 \
  --vae-patch-parallel-size 4 --vae-parallel-mode tile --vae-use-tiling \
  --diffusion-attention-backend FASTVIDEO_VSA

curl --fail-with-body -sS http://127.0.0.1:8095/v1/videos/sync \
  -F 'model=FastH3-V2' \
  -F 'prompt=A red fox runs through fresh snow at dawn, with fast pawsteps, winter wind and distant birds.' \
  -F 'width=1344' -F 'height=768' -F 'aspect_ratio=16:9' -F 'fps=24' \
  -F 'num_inference_steps=8' -F 'guidance_scale=1' -F 'flow_shift=10' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":4.4,"audio_flow_shift":3.0}' \
  -o fasth3_v2.mp4
```

V2 pins video/audio shifts **10/3**, guidance **1**, and the unshifted ladder
`[0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125, 0.0]`.
Use **`num_inference_steps=8`** in Omni: these nine nodes bound eight model
evaluations. FastVideo's `--steps 9` counts nodes instead. The existing H3 ODE
update is reused without stochastic re-noising.

The trained attention policy is **80% sparsity with 64-token video tiles**.
Omni derives the number of selected video tiles from each request's geometry;
do not supply a fixed `--fastvideo-vsa-topk`. Text/condition/audio prefix keys
remain available to every query, and prefix queries remain dense. Missing VSA
geometry or a failed H3 kernel raises an error instead of serving the student
with dense attention.

This release supports T2VA only, with local or pure Ulysses attention. Additional
LoRA adapters, Ref2VA and FL2VA conditioning are unsupported. The legacy
four-step adapter keeps its original sampling and fixed-top-k behavior.

## Key parameters

| Parameter | Recommended value | Notes |
| ----------- | ------------------- | ------- |
| `quality` | omitted or `lossless` | Request-level quality intent; `high` dynamically installs H3's conservative Cache-DiT profile |
| `extra_params.force_refresh_step_hint` | omitted | Optional positive 1-based denoising-step hint for an active Cache-DiT request; pair with `extra_params.force_refresh_step_policy`=`once` or `repeat` |
| `task` | `t2va`, `fl2va`, or `ref2va` | Passed in `extra_params`; selects the task-specific DiT |
| `duration` | Workload-specific | Decimal seconds in `extra_params`; converted to H3-compatible frame count |
| `fps` | `24` | H3 output FPS is fixed |
| `num_inference_steps` | `50` | Matches the reference accuracy workloads |
| `flow_shift` | `12` | Video sigma shift |
| `audio_flow_shift` | `3` | Audio sigma shift, passed in `extra_params` |
| `seed` | Task-specific | Use a fixed value for reproducibility |
| `aspect_ratio` | Task-specific | T2VA requires a named ratio; FL2VA follows the input image; Ref2VA defaults to `16:9` |
| `short_edge` | `768` | H3 shape policy requires exactly 768 when `width`/`height` are omitted |
| `num_outputs_per_prompt` | `1` | 1–10; async API returns every output |
| `start_time_seconds` | `0` | Reference-video segment start; use a list in `extra_params` for multiple videos |
| `width`, `height` | Multiples of 32 | Output aspect ratio must be between 1:4 and 4:1 |

## ComfyUI Frontend

Users can also use a ComfyUI frontend to interact with a hosted MiniMax-H3 service. The ComfyUI frontend can run in a separate environment or machine. Refer to [vLLM-Omni ComfyUI Integration](../../docs/features/comfyui.md) for details.

## Validated four-GPU evidence

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

## Validated FP8 evidence

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

## TeaCache acceleration

TeaCache reuses DiT block residuals across denoising steps when consecutive
timestep embeddings are similar. MiniMax-H3 TeaCache is currently calibrated
only for the FL2VA partition. In combined serving, FL2VA requests use TeaCache
while Ref2VA requests run uncached; Ref2VA-only serving rejects TeaCache.

TeaCache and Cache-DiT are mutually exclusive; pick one cache backend per server.

The model-specific default and examples use `rel_l1_thresh=0.17`, which
provided the best conservative speed/quality balance in the validated 107-frame
T2VA workload. Lower values
may produce few or no cache hits, while higher values can improve performance
at the cost of output quality. Validate the threshold on representative
prompts and generation settings before changing it.

### Offline (Python API)

Export the directory containing the `FL2VA` and `Ref2VA` subdirectories before
running the Python example, for example `export MODEL_ROOT=/models/MiniMax-H3`.

```python
import os

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model=os.path.join(os.environ["MODEL_ROOT"], "FL2VA"),
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.17},
    trust_remote_code=True,
    enable_cpu_offload=True,
)
outputs = omni.generate(
    "A quiet cinematic night scene with matching ambient sound.",
    OmniDiffusionSamplingParams(
        height=256,
        width=448,
        num_frames=29,
        fps=24,
        num_inference_steps=50,
        seed=42,
        extra_args={
            "task": "t2va",
            "duration": 4.0,
            "aspect_ratio": "16:9",
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        },
    ),
)
```

### Online serving

```bash
vllm serve "${MODEL_ROOT}/FL2VA" \
  --omni \
  --trust-remote-code \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh":0.17}'
```

## Known limitations

- TeaCache is calibrated for FL2VA only; Ref2VA requests run uncached.
- Combined serving requires sibling `FL2VA` and `Ref2VA` directories, loads
  both task-specific DiTs, and loads shared components once from `FL2VA`.
- Request mode executes one generation request per diffusion batch. Use
  `--step-execution` with `--max-num-seqs N` to admit several requests at once
  (see
  [Execution modes](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/execution_modes.md));
  step
  mode does not support `cache_backend`.
- The first regional-compile request is a warmup and should not be included in
  steady-state performance measurements.
- Ref2VA accepts up to 9 images, 3 video clips, and 3 audio clips, with at most
  12 references in total. At least one image or video is required; audio-only
  requests are rejected.
- The 768 px short-edge mode is available for T2VA and FL2VA; 1344x768 is the
  documented 16:9 request shape.
- `--cfg-parallel-size > 1` is rejected by design (CFG-distilled, no negative branch).
- VAE patch parallelism requires size 1 or the full DiT group size and supports the
  H3 native `tile` mode only.
- Ulysses x Ring hybrid attention supports H3's single-request contiguous suffix
  padding. Arbitrary attention masks and multi-request packed batches remain
  unsupported on the Ring path.
- Online FP8 with DLO AllGather temporarily materializes the complete FP8 model
  in host memory on every rank during startup before retaining only each rank's
  shard. Size startup host memory for that transient peak.
- TeaCache and Cache-DiT cannot be enabled on the same server.
- Pure Ulysses still replicates the full DiT on every rank, so smaller-memory GPUs
  cannot use `--usp N --tp 1` as a resident capacity path. Use DiT tensor parallelism
  or model-level CPU offload; text-encoder TP alone is not sufficient.

## Additional resources

- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
- [Diffusion parallelism](../../docs/user_guide/diffusion/parallelism/overview.md)
