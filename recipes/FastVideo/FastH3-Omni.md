# FastH3 Omni

[FastH3 family](README.md) · [Single-card guide](<FastH3 RefOmni Single Card.md>)

Run [FastH3 Omni](https://huggingface.co/FastVideo/FastVideo-FastH3-Omni-8-Step-V1)
on eight GPUs to generate video and audio from a prompt and reference image.
For RTX 4090, RTX 5090, DGX Spark, or RTX PRO 6000 Blackwell single-card
serving, use the [single-card guide](<FastH3 RefOmni Single Card.md>).

## Install

Follow the [CUDA installation guide](../../docs/getting_started/installation/gpu.md)
and install `ffmpeg`. From the vLLM-Omni repository root, install the
attention kernels used by this configuration:

```bash
uv pip install -e '.[cudnn-bsa]'
```

If Hugging Face requests access, accept the model's access terms and run
`hf auth login` once. The server downloads the required model components.

## Start the server

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve FastVideo/FastVideo-FastH3-Omni-8-Step-V1 --omni \
  --config recipes/FastVideo/fasth3-8step-8gpu.yaml
```

The configuration handles eight-GPU placement, eight inference steps, and
video decoding. Wait for `Application startup complete` before sending a
request. To change the default port, add `--port 8001`.

## Generate video and audio

Save your reference image as `reference.png` and run in a second terminal:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
  -F 'prompt=The person in the reference image smiles and waves at the camera. Birds chirp in the background.' \
  -F 'input_reference=@reference.png' \
  -F 'size=1344x768' -F 'seconds=5' \
  -o output.mp4
```

Open `output.mp4` to play the video with stereo audio. A five-second request
produces 124 frames at 24 fps. For a smaller output, use `size=832x480`.
For multiple images, video, or audio references, see the
[H3 reference API](../MiniMaxAI/MiniMax-H3.md).

## Optional NVFP4 video decoder

On Blackwell GPUs with FlashInfer installed, add this to the deployment
YAML to use the faster 8-layer decoder:

```yaml
additional-config:
  video_vae_decoder:
    model: FastVideo/FastH3-NVFP4-VAE-Decoder
    subfolder: fast-8L
```

Change `fast-8L` to `quality-36L` for the deeper decoder, then restart.
See the [decoder model card](https://huggingface.co/FastVideo/FastH3-NVFP4-VAE-Decoder)
for the available variants. For FP8 or NVFP4 QAD checkpoints, follow the
[single-card guide](<FastH3 RefOmni Single Card.md>).

## Benchmark: 8× NVIDIA RTX PRO 5000

Measured at 1344×768 with BF16 on eight 72 GB Blackwell GPUs, one request
at a time. Timing includes MP4 encoding and excludes model loading:

| Output | Generation time | Reported allocator peak |
| --- | ---: | ---: |
| 124 frames, about 5 seconds | 41.04 s median of 3 runs | 35.15 GiB |
| 345 frames, about 14 seconds | 95.93 s, 1 run | 38.94 GiB |

These are offline measurements, not HTTP response times. See the
[benchmark details](https://github.com/lishunyang12/vllm-omni/pull/63)
for the measured environment and configuration.
