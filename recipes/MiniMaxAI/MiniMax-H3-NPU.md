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
- CANN toolkit: 9.1.0
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

### Atlas 800I A2 / Atlas 800I A3

The recommended 8-NPU server command below uses Ulysses sequence parallelism
degree 8, text-encoder tensor parallelism degree 8, distributed layerwise
offload, and native tiled VAE patch parallelism degree 8.

#### Recommended configuration (Atlas 800I A2 / A3)

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT=9098
export MODEL=MiniMaxAI/MiniMax-H3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

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
  --diffusion-attention-config '{"default": {"backend": "RAINFUSION_ATTN",
      "block_sparse": {"sparsity": 0.8, "start_step": 12}}}'
```

Keep `--ring 1` when using RainFusion: the `rf_v2` kernel ranks key blocks
over the whole sequence, so ring parallelism would split away the keys it
needs. Scale with `--usp` instead.

When starting the Ref2VA service, replace the `--diffusion-attention-config`
flag above with:

```bash
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

#### Other optional performance optimizations

The following subsections list only the **incremental or replacement flags**
relative to the [Recommended configuration](#recommended-configuration-atlas-800i-a2--a3);
environment variables and all other server flags remain unchanged.

##### Enable Cache-DiT

Append the following flags to the end of the server command to enable DiT
block caching with TaylorSeer extrapolation:

```bash
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"max_warmup_steps":4,"residual_diff_threshold":0.4,"max_continuous_cached_steps":4,"enable_taylorseer":true,"taylorseer_order":2}'
```

##### Enable INT8 online quantization

Append `--diffusion-quantization-config '{"transformer":{"method":"int8"}}'`
after `--text-encoder-tp-size 8`, and append `--dlo-no-use-allgather` for
distributed layerwise offload:

```bash
  --diffusion-quantization-config '{"transformer":{"method":"int8"}}' \
  --dlo-no-use-allgather
```

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
PyTorch 2.10.0+cpu, and torch_npu 2.10.0.post2, using the recommended
configuration above:

| Workload | Configuration |
|----------|---------------|
| T2VA, 209 frames, 1344x768 | TE TP8, distributed layerwise offload, Ulysses 8, VPP8 tile, regional compile |
| Ref2VA (prompt + video), 124 frames, 1344x768 | TE TP8, distributed layerwise offload, Ulysses 8, VPP8 tile, regional compile |

These measurements describe the validated shapes rather than a general
throughput guarantee.

## Performance benchmark (Atlas 800I A3)

| NPUs | Workload | Parallelism | Precision | Input | Output | E2E (s) | Per step (s) |
| ---: | --- | --- | ---: | --- | --- | ---: | ---: |
| 8 | T2VA | TP8 + USP8, DLO | bf16 | prompt | 768P, 5s | 97.84 | 1.87 |
| 8 | T2VA | TP8 + USP8, DLO | bf16 | prompt | 768P, 10s | 245.77 | 4.75 |
| 8 | T2VA | TP8 + USP8, DLO | bf16 | prompt | 768P, 15s | 448.19 | 8.75 |
| 8 | FL2VA | TP8 + USP8, DLO | bf16 | prompt + first frame | 768P, 5s | 117.23 | 2.18 |
| 8 | FL2VA | TP8 + USP8, DLO | bf16 | prompt + first frame | 768P, 10s | 278.94 | 5.43 |
| 8 | FL2VA | TP8 + USP8, DLO | bf16 | prompt + first frame | 768P, 15s | 489.05 | 9.61 |
| 8 | Ref2VA | TP8 + USP8, DLO | bf16 | prompt + 5s reference video | 768P, 5s | 343.95 | 6.55 |
| 8 | Ref2VA | TP8 + USP8, DLO | bf16 | prompt + 10s reference video | 768P, 10s | 1088.93 | 21.40 |
| 8 | Ref2VA | TP8 + USP8, DLO | bf16 | prompt + 15s reference video | 768P, 15s | 2333.15 | 46.60 |

All the figures above were measured with the
[Recommended configuration](#recommended-configuration-atlas-800i-a2--a3).

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
