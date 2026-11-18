# FastH3 V2

[FastH3 family](README.md) · [FastH3 Omni](FastH3-Omni.md)

Generate video and audio from a text prompt with
[FastH3 V2](https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2).
No reference image is required. This eight-GPU deployment is experimental.
For reference-image generation, use [FastH3 Omni](FastH3-Omni.md).

## Install

Follow the [CUDA installation guide](../../docs/getting_started/installation/gpu.md)
and install `ffmpeg`. From the vLLM-Omni repository root:

```bash
uv pip install -e '.[cudnn-bsa]'
```

If Hugging Face requests access to MiniMax-H3 components, accept the model's
access terms and run `hf auth login` once.

## Start the server

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=7200 \
vllm serve FastVideo/FastVideo-FastH3-8-Step-V2 --omni \
  --config recipes/FastVideo/fasth3-8step-8gpu.yaml
```

The configuration handles eight-GPU placement, eight inference steps, and
video decoding. Wait for `Application startup complete` before sending a
request.

## Generate video and audio

In a second terminal:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
  -F 'prompt=A red fox runs through snow at dawn, with pawsteps, wind and distant birds.' \
  -F 'size=1344x768' -F 'seconds=5' \
  -o output.mp4
```

Open `output.mp4` to play the video with audio. For a smaller output, use
`size=832x480`. Keep the configuration's sampling defaults; this model is
trained for eight steps.
