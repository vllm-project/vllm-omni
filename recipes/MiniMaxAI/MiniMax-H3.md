# MiniMax H3

> Joint video and audio generation with text, first-frame, image/audio, or
> multi-video conditions

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Maintainer: Community

MiniMax H3 is a CFG-distilled joint video/audio diffusion transformer. Its
checkpoint has two independently served partitions:

- `FL2VA`: text-to-video+audio (`t2va`) and first-frame-to-video+audio
  (`fl2va`)
- `Ref2VA`: image+audio or one-or-more-video reference-to-video+audio
  (`ref2va`)

The generated MP4 contains H.264 video and synchronized stereo audio.

## Prerequisites

The checkpoint requires Hugging Face access approval. Download it and point
`MODEL_ROOT` at the directory containing the `FL2VA` and `Ref2VA`
subdirectories:

```bash
hf auth login
export MODEL_ROOT=/path/to/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --local-dir "${MODEL_ROOT}"
```

Install vLLM-Omni from the checkout containing MiniMax H3 support. On
Blackwell, install the optional FlashAttention-4 dependency:

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[fa4]'
```

`ffmpeg` and `ffprobe` must be available on `PATH`. They are used for
reference-video preparation and MP4 output.

## Start a server

One server loads one checkpoint partition. Set `MODEL` to `FL2VA` for T2VA
and FL2VA requests, or to `Ref2VA` for either Ref2VA request.

### Single GPU: accuracy and memory first

The single-GPU configuration uses model-level CPU offload.
This matches the accuracy-qualified reference path and prevents the Qwen3-VL
encoder and DiT from being resident on the GPU at the same time.

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --num-gpus 1 \
  --enable-cpu-offload \
  --diffusion-attention-backend FLASH_ATTN
```

Use a GPU with enough memory for the active H3 component and enough system RAM
for the offloaded components. CPU offload reduces GPU memory pressure but adds
PCIe/NVLink transfer latency.

### Four GPUs: measured best-practice throughput

The best validated four-GPU configuration on four NVIDIA B300 GPUs is:

- no CPU or layerwise offload;
- Ulysses sequence parallelism degree 4;
- native tiled VAE patch parallelism degree 4;
- regional `torch.compile` for the repeated DiT blocks;
- FlashAttention, with Ring and TP left at 1.

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1,2,3 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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
  --vae-use-tiling \
  --diffusion-attention-backend FLASH_ATTN
```

Do not add `--enforce-eager` to this performance configuration. The first
request includes regional compilation; warm the server once before measuring
steady-state latency. H3 is CFG-distilled, so `--cfg-parallel-size` must remain
1. The H3 VAE supports its native `tile` mode, not
`spatial_shard_height` or `spatial_shard_width`.

### Text encoder tensor parallelism

The Qwen3-VL text encoder (~51.5 GB in BF16 for the retained 50 layers) is by
default fully resident on the DiT main rank.  On multi-GPU no-offload runs that
rank becomes the peak-memory hotspot.  Add `--text-encoder-tp-size N` to shard
the encoder across the first `N` DiT ranks (the encoder is implemented with
vLLM-style tensor-parallel layers and runs with distributed collectives over
its own encoder process group):

```bash
export MODEL="${MODEL_ROOT}/FL2VA"
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1,2,3 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
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

`N` must divide the Qwen3-VL head counts (64 attention heads / 8 KV heads), so
valid values on a 4-GPU server are 1, 2, and 4 (1, 2, 4, 8 on 8 GPUs).  The
encoder TP rank set is the first `N` DiT ranks; on 4 GPUs
`--text-encoder-tp-size 4` shards the encoder 4-way, dropping the DiT main
rank's no-offload peak by roughly `(N-1)/N` of the ~51.5 GB encoder while the
other ranks each gain `~51.5/N` GB.  The encoder output remains identical to
the reference path within bf16 rounding: every encoder rank all-reduces the
row-parallel projections, so the full `[seq, 5120]` layer-50 hidden state is
replicated on every rank.

To serve Ref2VA, stop the FL2VA server and restart the same one- or four-GPU
command with:

```bash
export MODEL="${MODEL_ROOT}/Ref2VA"
```

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

Run this request against the `FL2VA` partition:

```bash
curl -sS -X POST "${API_URL}" \
  -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

### 2. FL2VA: first frame to video and audio

Run this request against the `FL2VA` partition. When width and height are
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

### 3. Ref2VA: image and audio reference

Run this request against the `Ref2VA` partition. `audio_reference` accepts an
HTTP(S) URL or a `data:` URL. In one terminal, expose the local reference
assets to the serving host:

```bash
python -m http.server 8092 \
  --bind 127.0.0.1 \
  --directory /path/to/reference_assets
```

Then submit the image and audio request from another terminal:

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

### 4. Ref2VA: two video references

Run this request against the `Ref2VA` partition. Repeat the
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

The server stores uploaded multi-video references only for the lifetime of the
request and deletes the temporary files after generation. Video Ref2VA uses
the source-video soundtracks and does not accept a separate
`audio_reference`.

## Key parameters

| Parameter | Recommended value | Notes |
|-----------|-------------------|-------|
| `task` | `t2va`, `fl2va`, or `ref2va` | Passed in `extra_params`; must match the served partition |
| `duration` | Workload-specific | Decimal seconds in `extra_params`; converted to H3-compatible frame count |
| `fps` | `24` | H3 output FPS is fixed |
| `num_inference_steps` | `50` | Matches the reference accuracy workloads |
| `flow_shift` | `12` | Video sigma shift |
| `audio_flow_shift` | `3` | Audio sigma shift, passed in `extra_params` |
| `seed` | Task-specific | Use a fixed value for reproducibility |
| `width`, `height` | Multiples of 32 | Aspect ratio must be between 1:4 and 4:1 |

## Validated four-GPU evidence

The four-GPU recommendation was measured on four NVIDIA B300 GPUs with one
excluded warmup followed by three requests.

| Workload | Configuration | Observed result |
|----------|---------------|-----------------|
| FL2VA, 209 frames, 1248x768 | no offload, U4, VPP4 tile, regional compile | 86.964 s mean HTTP client latency |
| Two-video Ref2VA, 362 frames, 1344x768 | no offload, U4, VPP4 tile, regional compile | 784.394 s accounted model-stage mean |

These measurements describe the validated shapes rather than a general
throughput guarantee. Multi-video Ref2VA is much slower because the two
reference videos expand both the Qwen3-VL vision sequence and the packed DiT
attention sequence.

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

- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
- [Diffusion parallelism](../../docs/user_guide/diffusion/parallelism/overview.md)
