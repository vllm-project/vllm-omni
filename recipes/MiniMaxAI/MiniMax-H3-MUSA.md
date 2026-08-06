# MiniMax H3 on Moore Threads MUSA

> Joint video and audio generation on Moore Threads GPUs

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Hardware: 4x Moore Threads MTT S5000 for the validated Ref2VA profile
- Maintainer: Community

This recipe adapts [MiniMax-H3.md](MiniMax-H3.md) for MUSA environments.
MiniMax H3 has two independently served checkpoint partitions:

- `FL2VA` serves text-to-video+audio (`t2va`) and
  first-frame-to-video+audio (`fl2va`).
- `Ref2VA` serves image/audio or video-reference generation (`ref2va`).

## Prerequisites

### Checkpoint

Only download the partition needed by the server. Downloading the whole model
repository also retrieves duplicate shared components and the original
top-level pipeline layout.

For T2VA and FL2VA:

```bash
python -m pip install modelscope
export MODEL_ROOT=/path/to/MiniMax-H3
modelscope download MiniMax/MiniMax-H3 \
  --local_dir "${MODEL_ROOT}" \
  --max-workers 16 \
  --include 'FL2VA/**'
```

For Ref2VA, download `Ref2VA/**` instead.

### Environment

Install a compatible PyTorch/torch-musa, torchada, MATE, and vLLM-MUSA
stack before installing vLLM-Omni. Importing vLLM-Omni should report that the
`musa` vLLM platform plugin is active.

Install vLLM-Omni from a checkout containing MiniMax H3 support:

```bash
python -m pip install -e .
```

MiniMax H3 reference inputs and MP4 output require `soundfile`, `ffmpeg`, and
`ffprobe`. The Python dependency is installed by vLLM-Omni; install the two
executables with the operating system package manager and verify them:

```bash
python -c 'import soundfile; print(soundfile.__version__)'
ffmpeg -version
ffprobe -version
```

When torchaudio cannot load TorchCodec, vLLM-Omni automatically falls back to
soundfile. Formats that libsndfile cannot read are demuxed through ffmpeg.

## Start a server

One server loads one checkpoint partition. Set `MODEL` to `FL2VA` for T2VA
and FL2VA, or to `Ref2VA` for Ref2VA.

The validated Ref2VA configuration uses tensor parallelism across four MTT
S5000 GPUs and offloads inactive model components to CPU:

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

MUSA_VISIBLE_DEVICES=0,1,2,3 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 4 \
  --tensor-parallel-size 4 \
  --vae-patch-parallel-size 4 \
  --enable-cpu-offload \
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

On MUSA, `FLASH_ATTN` selects the MATE/FlashAttention-3 implementation. Do
not install the CUDA-only `fa4` optional dependency.

For request examples and parameter definitions, see
[MiniMax-H3.md](MiniMax-H3.md#http-api-examples).

## MUSA validation

The following weight-independent gates have been validated on one MTT S5000:

- packed variable-length attention selects MATE FlashAttention-3 and returns
  finite BF16 output;
- MiniMax H3 RoPE casts CPU FP64 position metadata to FP32 before device
  arithmetic and returns finite output;
- the video-VAE seeded RNG context is deterministic on MUSA and restores both
  CPU and MUSA RNG states;
- the soundfile fallback preserves stereo channel layout, FP32 samples, and
  the native sample rate when torchaudio/TorchCodec is unavailable.

Ref2VA image+audio generation was also validated end to end on one MTT S5000
with CPU offload and VAE tiling. The smoke used a 448x256 reference image, a
four-second 32 kHz stereo audio reference, two denoising steps, and seed 42.
The synchronous HTTP request completed in 50.6 seconds and returned an MP4
with the following decoded properties:

- 107 video frames at 448x256 and 24 FPS;
- non-static video (`pixel_std=13.07`, `temporal_delta=0.49`);
- finite stereo audio at 32 kHz (`audio_rms=0.0458`).

Official Ref2VA video-reference serving was validated with the expanded
MiniMax prompt at 1344x768, 5 seconds, seed 0, and 50 steps. A true TP4 run on
four MTT S5000 GPUs returned HTTP 200 in 638.6 seconds and produced 124 H.264
frames at 24 FPS with AAC stereo audio at 32 kHz.

Keep `--vae-use-tiling` enabled for this serving profile.

## Known limitations

- Keep Ring Attention at degree 1. The current Ring path does not preserve
  MiniMax H3 packed padding boundaries.
- H3 is CFG-distilled, so `--cfg-parallel-size` must remain 1.
- Each server process loads only one checkpoint partition.
- H3 currently executes one generation request per diffusion batch.
- FP8 quantization has not been enabled for MiniMax H3.
- MP3, M4A, MP4, and reference-video audio fallback requires `ffmpeg` on
  `PATH`; WAV inputs can be read directly through soundfile.

## Additional resources

- [MiniMax-H3.md](MiniMax-H3.md) — full model and API guide
- [MiniMax-H3-NPU.md](MiniMax-H3-NPU.md) — Ascend NPU deployment guide
- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
