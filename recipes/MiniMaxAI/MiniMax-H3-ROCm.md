# MiniMax-H3 on AMD ROCm (gfx942 / gfx950)

[Model guide](MiniMax-H3.md) · [Deployment choices](MiniMax-H3.md#choose-a-deployment) · [HTTP API](MiniMax-H3.md#http-api-examples)

MiniMax H3 runs on AMD Instinct GPUs (gfx942 / gfx950) in BF16. Use
`--diffusion-attention-backend FLASH_ATTN`, which resolves to AITER packed varlen
attention on both architectures.

## Installation

Complete the model guide's [checkpoint prerequisites](MiniMax-H3.md#prerequisites).

Install without the CUDA-only `[fa4]` extra. The VAE uses AITER GroupNorm on ROCm.

Install (ROCm wheel + source vLLM-Omni):

```bash
pip install "vllm==0.26.0+rocm723" \
  --extra-index-url https://wheels.vllm.ai/rocm/0.26.0/rocm723
VLLM_OMNI_TARGET_DEVICE=rocm pip install -e . --no-build-isolation
```

Prebuilt image: `vllm/vllm-omni-rocm:minimax-h3`. All tasks work out of the box:
the image bundles TorchCodec (for image+audio Ref2VA) and `ffmpeg` (for
video-reference Ref2VA).

## ROCm single GPU

Single GPU with model-level CPU offload keeps the Qwen3-VL encoder and DiT from
being co-resident:

```bash
vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni --trust-remote-code \
  --num-gpus 1 --enable-cpu-offload \
  --diffusion-attention-backend FLASH_ATTN
```

This configuration stages the VAEs as well as the encoder and DiT.
Its compact-API equivalent selects `dit`, `text_encoder`, and `vae` with
`mode: module`.

## ROCm four GPUs

Use Ulysses sequence parallelism, text-encoder TP4, and tiled VAE patch
parallelism without CPU offload. The text encoder is sharded across four
ranks; each rank retains a full DiT replica.

```bash
vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni \
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

## Validated ROCm evidence

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
