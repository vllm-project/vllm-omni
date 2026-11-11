# FastH3 RefOmni Single Card

[FastH3 family](README.md) · [Multi-GPU serving](FastH3-Omni.md)

Turn a reference image and a text prompt into a video with stereo audio.
Choose the command for your GPU below; the configuration handles memory
placement, attention, eight inference steps, and video decoding.

| GPU | Default checkpoint | Configuration |
| --- | --- | --- |
| RTX 5090 (32 GB), DGX Spark (128 GB), RTX PRO 6000 Blackwell (96 GB) | NVFP4 QAD | [Blackwell YAML](fasth3-omni-quantized.yaml) |
| RTX 4090 (24 GB) | BF16 with online FP8 | [RTX 4090 YAML](fasth3-omni-rtx4090.yaml) |

## 1. Install

Use Linux and Python 3.12. Install vLLM and PyTorch following the
[CUDA installation guide](../../docs/getting_started/installation/gpu.md),
then run from the vLLM-Omni repository root (system packages shown for Ubuntu):

```bash
uv pip install -e .
sudo apt-get update
sudo apt-get install -y ffmpeg
```

On DGX Spark, use the native `aarch64` environment. Blackwell uses the
FlashInfer package supplied by the matching vLLM installation.
If Hugging Face requests access to the MiniMax-H3 components, accept the
model's access terms and run `hf auth login` once.

## 2. Start the server

**RTX 5090, DGX Spark, or RTX PRO 6000 Blackwell:**

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve FastVideo/FastH3-Omni-NVFP4-QAD --omni \
  --config recipes/FastVideo/fasth3-omni-quantized.yaml
```

This loads the smaller prequantized checkpoint and uses the fast 8-layer
video decoder. Layer offload reduces GPU memory use.

**RTX 4090:**

```bash
uv pip install -e '.[vsa]'

CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --omni \
  --config recipes/FastVideo/fasth3-omni-rtx4090.yaml
```

The first start downloads weights; the 4090 also converts eligible weights
to FP8 while loading. Subsequent starts reuse the download cache.
Keep the server terminal open and wait for `Application startup complete`.

## 3. Generate video and audio

In a second terminal, save your image as `reference.png` and run:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
  -F 'prompt=The person in the reference image smiles and waves at the camera. The camera slowly moves closer. Birds chirp as the person says, "Welcome to the garden!"' \
  -F 'input_reference=@reference.png' \
  -F 'size=832x480' \
  -F 'seconds=5' \
  -F 'seed=1234' \
  -o output.mp4
```

Open `output.mp4`: it contains 124 frames at 24 fps and 32 kHz stereo audio.
For 768P, change `size=832x480` to `size=1344x768`.
For multiple images, video, or audio references, see the
[H3 reference API](../MiniMaxAI/MiniMax-H3.md).

## Optional settings

- **FP8 checkpoint:** in the Blackwell command, replace the model ID with
  `FastVideo/FastH3-Omni-FP8`. A local snapshot directory also works.
  The loader detects the format; this export's NVFP4 text encoder also
  requires Blackwell.
- **36-layer decoder:** in the Blackwell YAML, change
  `additional-config.video_vae_decoder.subfolder` from `fast-8L` to
  `quality-36L`, then restart. Both come from
  [FastH3-NVFP4-VAE-Decoder](https://huggingface.co/FastVideo/FastH3-NVFP4-VAE-Decoder).
- **DGX Spark:** remove `diffusion-offload-config` from the Blackwell YAML
  to keep the model resident in unified memory and avoid offload transfers.
- **Memory is tight:** start at 480P, keep the default decoder and tiled
  decoding, and close other GPU workloads.
