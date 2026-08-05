# MiniMax H3 on Ascend NPU

> Joint video and audio generation with text, first-frame, image/audio, or
> multi-video conditions — Ascend NPU deployment guide

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

Same as the GPU recipe — Hugging Face access approval is required:

```bash
hf auth login
export MODEL_ROOT=/path/to/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --local-dir "${MODEL_ROOT}"
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
operators (`adalayernorm`, etc.):

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

One server loads one checkpoint partition. Set `MODEL` to `FL2VA` for T2VA
and FL2VA requests, or to `Ref2VA` for Ref2VA requests.

### Multi-NPU: 768P validated configuration

Validated on eight NPUs of an Atlas 800I A3 server with Ulysses sequence
parallelism degree 8, text-encoder tensor parallelism degree 8, native tiled
VAE patch parallelism degree 8, and layerwise offload:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT=9098
export MODEL="${MODEL_ROOT}/FL2VA"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --enable-layerwise-offload \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --diffusion-attention-backend FLASH_ATTN
```

Do not add `--enforce-eager`. The first request includes regional
compilation; warm the server once before measuring steady-state latency.
H3 is CFG-distilled, so `--cfg-parallel-size` must remain 1.

To serve Ref2VA, stop the FL2VA server and restart with:

```bash
export MODEL="${MODEL_ROOT}/Ref2VA"
```

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
| T2VA, 209 frames, 1344x768 | TE TP8, layerwise offload, Ulysses 8, VPP8 tile, regional compile |
| Ref2VA (prompt + video), 124 frames, 1344x768 | TE TP8, layerwise offload, Ulysses 8, VPP8 tile, regional compile |

These measurements describe the validated shapes rather than a general
throughput guarantee.

## Known limitations

- Each server process loads only one checkpoint partition.
- H3 currently executes one generation request per diffusion batch.
- The first regional-compile request is a warmup and should not be included
  in steady-state performance measurements.
- Image+audio Ref2VA accepts exactly one image and one audio reference.
- Video Ref2VA accepts one or more video files, but not an additional
  standalone audio reference.
- VAE patch parallelism requires size 1 or the full DiT group size and
  supports the H3 native `tile` mode only.

## Additional resources

- [MiniMax-H3.md](MiniMax-H3.md) — full GPU guide
- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
