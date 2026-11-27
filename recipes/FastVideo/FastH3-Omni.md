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

## Request-level LoRA adapters

For a distributed-layerwise-offload deployment, add `--lora-backend peft` to enable ordinary H3 adapters, including Character Swap. Keep the checkpoint's eight-step sampling settings and disable Turbo adapters.

In ComfyUI, chain existing LoRA nodes through their `lora` input and set each node's strength independently. The HTTP API accepts the corresponding `lora` list; see [stacking multiple LoRAs](../../docs/user_guide/diffusion/lora.md#peft-backend-stacking-multiple-loras-in-one-request). This support applies to FastH3 Omni, not FastH3 Preview or V2.

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

## Editing, continuation, timeline guides and ControlNet

FastH3 Omni serves Ref2VA only, so every request below sends `"task":"ref2va"`
and at least one reference image (a `continue_from` request may omit it). The
eight-step schedule, `guidance_scale=1` and the flow shifts stay pinned.

- **Latent-mask editing:** send `source_video` and `video_noise_mask` as in
  [latent-mask editing](../../docs/serving/videos_api.md#minimax-h3-latent-mask-editing).
- **Continuation:** use `continue_from` with a completed job's id
  ([continuation across requests](../../docs/serving/videos_api.md#minimax-h3-continuation-across-requests)),
  or `"long_video":true` for in-request latent-tail continuation. As with the base
  Ref2VA checkpoint, the hidden tail is followed loosely: expect a mild jump at the
  join, and later windows may drift back toward the reference composition.
- **Timeline guides:** send `timeline_guides` and `guide_files` as in
  [timeline guides](../../docs/serving/videos_api.md#minimax-h3-timeline-guides),
  with `quality=lossless`.

### ControlNet

The [Fun ControlNet Union](../MiniMaxAI/MiniMax-H3.md#fun-controlnet-union-ctrl-01)
branch runs on FastH3 Omni with tensor parallelism instead of Ulysses and HSDP.
Copy `fasth3-8step-8gpu.yaml`, set `usp: 1`, delete `use-hsdp: true`, and add
`tensor-parallel-size: 8`, then start the server with
`--controlnet-model-path /path/to/MiniMax-H3-Fun-Controlnet-Union.safetensors`.
Control requests are Ref2VA requests with a reference image:

```bash
curl --fail-with-body -sS http://localhost:8000/v1/videos/sync \
  -F 'prompt=A snowy winter evening at the school gate.' \
  -F 'input_reference=@reference.png' \
  -F 'control_type=canny' -F 'control_reference=@canny.mp4' \
  -F 'extra_params={"task":"ref2va","canny":{"control_context_scale":1.0}}' \
  -F 'size=1344x768' -F 'num_frames=124' -F 'quality=lossless' \
  -o controlled.mp4
```

Control describes only the target video; reference rows receive no control
hint. Control cannot be combined with `continue_from`, long-video continuation,
timeline guides or latent-mask editing, and needs the dense BF16 checkpoint (the
NVFP4 QAD export is rejected). The control weights were trained against the base FL2VA transformer, so
control strength and fidelity on the distilled checkpoint may differ from base H3.

### Measured on 8x H200

One 1344x768, 124-frame request per feature, seed 20261122, vLLM 0.34.0.
Editing and continuation ran on the Ulysses 8 + HSDP configuration above; timeline
guides and ControlNet ran on the TP 8 control configuration:

| Feature | Latency | Check |
| --- | --- | --- |
| Ref2VA baseline | 12-13 s warm | - |
| Latent-mask edit, right half regenerated | 18.1 s | kept half PSNR 33.2 dB vs source; regenerated half 20.5 dB |
| `continue_from` (119 new frames) | 12.0 s | SSIM across the join 0.62; adjacent frames within a clip average 0.81 |
| 20 s `long_video` continuation (481 frames) | 40.3 s | SSIM across the window boundary 0.91 |
| Timeline guide at frame 96 | 21.3 s | SSIM to the guide image 0.85 at frame 96 (0.25 without guides); best-matching frame 96 |
| ControlNet canny, strength 1.0 | 27.6 s | edge F1 vs control 0.85 (0.44 at strength 0) |

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
