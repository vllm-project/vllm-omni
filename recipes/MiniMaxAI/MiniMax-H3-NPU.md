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
- TorchNpu: 2.10.0.post2
- vLLM-Omni 安装：

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

- `ffmpeg` / `ffprobe` 必须在 `PATH` 上（参考视频准备与 MP4 输出使用）。
- `decord` 
- 音频输入无需 TorchCodec：wav/mp3/m4a/mp4 均通过 soundfile / ffmpeg
  fallback 以原生采样率加载。

## Start a server

One server loads one checkpoint partition. Set `MODEL` to `FL2VA` for T2VA
and FL2VA requests, or to `Ref2VA` for Ref2VA requests.

### Multi-NPU: 768P validated configuration

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
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

To serve Ref2VA, stop the FL2VA server and restart with:

```bash
export MODEL="${MODEL_ROOT}/Ref2VA"
```

## HTTP API examples

请求方式与 GPU 版完全一致，见
[MiniMax-H3.md § HTTP API examples](MiniMax-H3.md#http-api-examples)。
注意在 NPU 环境下使用 **768P 规格**（如 `width=1344 height=768`）。

## Key parameters

与 GPU 版一致，见 [MiniMax-H3.md § Key parameters](MiniMax-H3.md#key-parameters)。
NPU 环境额外约束：

| Parameter | Recommended value | Notes |
|-----------|-------------------|-------|
| `width`, `height` | 短边 ≤ 768（如 1344x768） | 2K 分辨率当前会 OOM |

## Validated NPU evidence

| Workload | Configuration | Observed result |
|----------|---------------|-----------------|
| T2VA, 209, 1344x768 | Text Encoder TP8, Dit offload + U8, VPP8 tile, regional compile | 480s |
| Ref2VA（prompt+video）, 124, 1344x768 | Text Encoder TP8, Dit offload + U8, VPP8 tile, regional compile | 980s |

- 验证环境：800I A3 x 8，CANN 9.0.1 /torch 2.10.0+cpu /torch_npu 2.10.0.post2

## Known limitations

- Each server process loads only one checkpoint partition.
- H3 currently executes one generation request per diffusion batch.
- The first regional-compile request is a warmup and should not be included in
  steady-state performance measurements.
- Image+audio Ref2VA accepts exactly one image and one audio reference.
- Video Ref2VA accepts one or more video files, but not an additional standalone
  audio reference.
- VAE patch parallelism requires size 1 or the full DiT group size and supports
  the H3 native `tile` mode only.


## Additional resources

- [MiniMax-H3.md](MiniMax-H3.md)（GPU 版完整指南）
- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
