# MiniMax H3

> Joint video and audio generation with text, first/last keyframes, and
> mixed image/video/audio references

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Maintainer: Community

MiniMax H3 is a CFG-distilled joint video/audio diffusion transformer. Its
checkpoint has two task-specific DiT partitions:

- `FL2VA`: text-to-video+audio (`t2va`) and first-frame-to-video+audio
  (`fl2va`)
- `Ref2VA`: up to 9 images, 3 videos, and 3 audio references in supported
  image/video/audio combinations (`ref2va`; audio-only is rejected)

One vLLM-Omni diffusion stage can load both DiTs while instantiating the
tokenizer, processor, Qwen3-VL text encoder, video VAE, and audio VAE only
once. Requests select the DiT with `extra_params.task`.

The generated MP4 contains H.264 video and synchronized stereo audio.

## Prerequisites

The checkpoint requires Hugging Face access approval. Authenticate once;
`vllm serve` downloads the required components automatically:

```bash
hf auth login
export MODEL=MiniMaxAI/MiniMax-H3
```

The vLLM-Omni pipeline downloads `FL2VA/**`, `Ref2VA/model_index.json`, and
`Ref2VA/transformer/**`. It does not download or load the diffusers-format
`transformer`, `transformer_ref`, or `vae` weights at the repository root, nor
duplicate Ref2VA copies of shared components.

Install vLLM-Omni from the checkout containing MiniMax H3 support. The
two-GPU RTX 5090/4090 profiles use cuDNN attention and do not need
FlashAttention-4. Install the optional dependency only for the four-GPU
B300/GB200 `FLASH_ATTN` profile:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

To keep FA4 available as an explicit option on Blackwell, install the
FlashAttention-4 extra:

```bash
uv pip install -e '.[fa4]'
```

`ffmpeg` and `ffprobe` must be available on `PATH`. They are used for
reference-video preparation and MP4 output.

## Start a server

Pass the repository ID directly. The pipeline uses `FL2VA` for model discovery
and shared components, and loads the second DiT from `Ref2VA/transformer`.

### Memory and storage requirements

Treat GPU HBM, host RAM, and checkpoint storage as separate requirements. Each
H3 checkpoint partition (`FL2VA` or `Ref2VA`) contains about **134 GiB** of
BF16 safetensors (about **135 GiB** on disk). Keeping both partitions
locally therefore needs roughly **270 GiB** of model storage. A combined
service downloads both; `--task-type fl2va` or `--task-type ref2va` downloads
only the selected partition.

CPU offload and distributed layerwise offload reduce GPU residency; they do
not make the model weights disappear. With `--dlo-no-use-allgather`, each
worker retains its standard-loader rank-local weights in host memory, including
pinned CPU buffers used for H2D streaming. Use at least **200 GiB available
system RAM** before starting the two-GPU recipe; a **384 GiB host is
recommended** to leave room for the OS, CUDA/PyTorch allocations, request
inputs, and filesystem cache. Do not run the FL2VA and Ref2VA servers at the
same time on a host sized for this minimum.

The consumer-GPU profiles below are HBM budgets only. They still require the
host-RAM budget above.

### Single GPU: accuracy and memory first

The single-GPU configuration uses model-level CPU offload.
This matches the accuracy-qualified reference path and prevents the Qwen3-VL
encoder and DiT from being resident on the GPU at the same time.

```bash
export MODEL=MiniMaxAI/MiniMax-H3
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
for both offloaded DiTs plus the shared components. Model-level offload keeps
the two DiTs mutually exclusive on GPU, but adds PCIe/NVLink transfer latency.

### Two 24/32 GB GPUs: TP2 distributed layerwise offload

For two PCIe consumer GPUs, combine TP2 with distributed layerwise offload
(DLO). The standard loader first creates the rank-local TP shard. DLO keeps
that shard in pinned host memory and streams the 30 tail DiT blocks through a
shared two-buffer window without a DP AllGather. The first 20 DiT blocks are
copied to the GPUs once per denoise stage, reused by every sampling step, and
released before VAE decode so the decoder can reuse their HBM.

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 \
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 2 \
  --tensor-parallel-size 2 \
  --usp 1 \
  --ring 1 \
  --text-encoder-tp-size 2 \
  --vae-patch-parallel-size 2 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --enable-distributed-layerwise-offload \
  --dlo-no-use-allgather \
  --dlo-resident-layers 20 \
  --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

Use the profile that matches the per-GPU memory capacity:

| Profile | GPUs | Starting shape | Resident DiT blocks | Attention | Execution | Status |
|---|---:|---:|---:|---|---|---|
| `rtx5090` | 2 x 32 GB | 1344x768 | 20 | cuDNN attention | eager | Target-hardware validated |
| `rtx4090` | 2 x 24 GB | 1024x576 | 12 | cuDNN attention | eager | Capacity-proxy starting point |

This topology uses all available parallel capacity: TP2 shards both the DiT
and text encoder, `--dlo-no-use-allgather` streams each rank's local TP shard
without reconstructing full blocks, and VAE patch parallelism splits tiled
decode across both GPUs. cuDNN attention is selected explicitly for the RTX
consumer path; the server stays eager to avoid an unqualified compile path.

The resident count changes placement and transfer frequency only; it does not
quantize or change the BF16/FP32 denoise math. Re-measure peak memory before
increasing it on a different request shape.

### RTX 5090 target-hardware validation

At vLLM-Omni commit `ae6577ea`, one full 50-step T2VA request completed on
2 x RTX 5090 without OOM:

| Shape | Frames | Client E2E | Sampled peak/GPU | Output validation |
|---:|---:|---:|---:|---|
| 1344x768 | 124 at 24 FPS | 8 min 38 s | approximately 22.6 GiB | H.264 video + 32 kHz stereo AAC; full `ffmpeg` decode passed |

This is a single end-to-end validation run, not a warmed multi-run latency
benchmark. The sampled `nvidia-smi` peak is also not a CUDA allocator
high-water mark. The environment used vLLM 0.26.0, vLLM-Omni
`0.26.1.dev14+gae6577ea`, and PyTorch 2.11.0+cu130. The
[run record](https://github.com/lishunyang12/vllm-omni-rankings/blob/dcd06d7e83cb069842535918c0169ee9f3f29ba0/scripts/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20260805000034_86_237.png)
captures the environment, output contract, elapsed time, and sampled peak.

Before the target run, both profiles were exercised on two B300 ranks as an
allocation and correctness proxy. At 1344x768, 124 frames, and 50 steps, the
20-layer profile peaked at 27,726 MiB per rank. At 1024x576, the 12-layer
profile peaked at 18,888 MiB per rank in a 5-step capacity run. The resident
and fully streamed placements produced identical decoded video-frame and audio
hashes for the same shape, step count, prompt, and seed. The B300 result does
not establish RTX 4090 PCIe latency; treat the 4090 profile as a conservative
starting point until it is measured on that GPU.

To run T2VA, FL2VA, image+audio Ref2VA, and two-video Ref2VA in order, validate
every MP4's H.264/AAC streams, and retain live server and GPU-memory logs:

```bash
RUN_ROOT=/path/to/run-root \
MODEL_ROOT=/path/to/MiniMax-H3 \
GPU_IDS=0,1 \
PROFILE=rtx5090 \
bash examples/offline_inference/minimax_h3/run_h3_2gpu_all_tasks.sh
```

The script selects 20 resident layers for `PROFILE=rtx5090` and 12 for
`PROFILE=rtx4090`; `DLO_RESIDENT_LAYERS=N` overrides either default.

### Four GPUs: throughput-oriented combined service

For a combined service on four high-memory GPUs, use:

- no CPU or layerwise offload;
- Ulysses sequence parallelism degree 4;
- native tiled VAE patch parallelism degree 4;
- regional `torch.compile` for the repeated DiT blocks;
- dense BF16 `TRTLLM_ATTN`, with Ring and TP left at 1.

Both DiTs remain resident in this no-offload configuration. If they do not fit,
use model-level CPU offload.

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1,2,3 \
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
  --vae-use-tiling
```

Do not add `--enforce-eager` to this performance configuration. The first
request includes regional compilation; warm the server once before measuring
steady-state latency. H3 is CFG-distilled, so `--cfg-parallel-size` must remain
1. The H3 VAE supports its native `tile` mode, not
`spatial_shard_height` or `spatial_shard_width`.

### Attention Backends

On datacenter Blackwell GPUs, MiniMax H3 defaults to dense BF16
`TRTLLM_ATTN`; no attention backend flag is required. To select it explicitly,
use:

```bash
--diffusion-attention-backend TRTLLM_ATTN
```

Stable measurements with the four-GPU profile above put dense `TRTLLM_ATTN`
and FA4 within 2% of each other. `TRTLLM_ATTN` remains the datacenter Blackwell
default and enables the optional optimizations below. Confirm the server log
contains `Defaulting to diffusion attention backend TRTLLM_ATTN` before
recording measurements when using the default selection.

FA4 remains available by explicitly selecting the `FLASH_ATTN` backend:

```bash
--diffusion-attention-backend FLASH_ATTN
```

On Blackwell, `FLASH_ATTN` selects FA4. Confirm the server log contains
`Using CuTe FlashAttention-4 on Blackwell` before recording FA4 measurements.

`TRTLLM_ATTN` additionally supports two **lossy** optimizations for the long main
DiT attention sequence: SAGE attention quantization and Skip-Softmax Sparse
Attention. SAGE quantizes Q/K to the configured dtype and V to FP8. This example uses
`fp8_e4m3` for Q/K; B200 also supports `int8` Q/K. The TRTLLM SAGE path fixes V
to FP8, so vLLM-Omni only exposes the Q/K dtype. The token refiner is a short
attention path, so the `per_role` override leaves SAGE and Skip-Softmax disabled
for it. The example enables the calibration-free Skip-Softmax path with
`threshold=0.05`, after the normalized timestep reaches `0.97`:

```bash
--diffusion-attention-config '{
  "default": {
    "backend": "TRTLLM_ATTN",
    "quant": {
      "dtype_qk": "fp8_e4m3",
      "q_block_size": 1,
      "k_block_size": 16
    },
    "skip_softmax": {
      "threshold": 0.05,
      "disabled_until_timestep": 0.97
    }
  },
  "per_role": {
    "minimax_h3.token_refiner": {
      "backend": "TRTLLM_ATTN"
    }
  }
}'
```

For configuration details, see
[TRTLLM_ATTN Backend and Skip-Softmax](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends.md#trtllm_attn-backend-and-skip-softmax)
and
[TRTLLM_ATTN SAGE Quantization](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/diffusion/attention_backends.md#trtllm_attn-sage-quantization).

### Text encoder tensor parallelism

The Qwen3-VL text encoder (~51.5 GB in BF16 for the retained 50 layers) is by
default fully resident on the DiT main rank.  On multi-GPU no-offload runs that
rank becomes the peak-memory hotspot.  Add `--text-encoder-tp-size N` to shard
the encoder across the first `N` DiT ranks (the encoder is implemented with
vLLM-style tensor-parallel layers and runs with distributed collectives over
its own encoder process group):

```bash
export MODEL=MiniMaxAI/MiniMax-H3
export PORT=8091

CUDA_VISIBLE_DEVICES=0,1,2,3 \
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
  --vae-use-tiling
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

No restart is needed: `task=fl2va` routes to `FL2VA/transformer`, while
`task=ref2va` routes to
`Ref2VA/transformer`. T2VA uses the FL2VA DiT.

### Online FP8 quantization

MiniMax H3 supports load-time FP8 quantization of the DiT. The checkpoint
remains BF16 on disk; vLLM-Omni quantizes eligible weights while loading and
uses dynamic activation scaling during inference. By default, attention and
MLP linears in the token refiner and main DiT blocks, the condition
projection, and all AdaLN projections use FP8. Patch, timestep, and final
projections remain FP32; the text encoder and VAEs are unchanged.

Add this option to an existing H3 server command:

```bash
--quantization fp8
```

Use `ignored_layers` to keep any otherwise eligible linear in BF16. H3
resolves the `transformer` component before constructing the DiT, so names do
not start with `transformer.`. Entries are exact runtime linear prefixes; a
parent name such as `blocks.0.attn` does not exclude its children.

Eligible names are the `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and
`mlp.fc2` children under `token_refiner.blocks.<0-1>` or `blocks.<0-49>`.
The other eligible names are `condition_proj`,
`blocks.<0-49>.adaln_proj.linear`, and `final_layer.adaln_proj.linear`.
For example, keep the first main block's attention projections in BF16 with:

```bash
--diffusion-quantization-config \
  '{"transformer":{"method":"fp8","ignored_layers":["blocks.0.attn.qkv_proj","blocks.0.attn.out_proj"]}}'
```

The structured option replaces `--quantization fp8`. Online FP8 is currently
incompatible with H3 layerwise offload because the offload path produces a
weight stride rejected by the Cutlass FP8 kernel. Use resident FP8 with tensor
parallelism and VAE tiling instead.

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

Run this request against the combined service:

```bash
curl -sS -X POST "${API_URL}" \
  -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":8.7,"audio_flow_shift":3.0}' \
  -o t2va.mp4
```

### 2. FL2VA: first frame to video and audio

Run this request against the combined service. When width and height are
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

Use the same `FL2VA` partition for the official tail-keyframe forms. A single
image with `frame_indices=[-1]` conditions the last frame; two ordered images
with `frame_indices=[0,-1]` condition the first and last frames:

```bash
export LAST_FRAME=/path/to/fl2va_last_frame.png
export FIRST_FRAME=/path/to/fl2va_first_frame.png

curl -sS -X POST "${API_URL}" \
  -F 'prompt=The subject moves naturally from the first image to the last image.' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=2102' \
  -F 'extra_params={"task":"fl2va","duration":8.7,"frame_indices":[0,-1],"audio_flow_shift":3.0}' \
  -F "input_references=@${FIRST_FRAME};type=image/png" \
  -F "input_references=@${LAST_FRAME};type=image/png" \
  -o fl2va_first_last.mp4
```

### 3. Ref2VA: image-only, image/audio, or mixed references

Run these requests against the combined service or a Ref2VA-only service.
Image-only Ref2VA omits `audio_reference`; adding one or more audio references
is optional. The typed fields accept one object or an ordered JSON list.
`audio_reference` accepts an HTTP(S) URL or a `data:` URL. In one terminal,
expose the local reference assets to the serving host:

```bash
python -m http.server 8092 \
  --bind 127.0.0.1 \
  --directory /path/to/reference_assets
```

Then submit an image-only request from another terminal:

```bash
export REF_IMAGE=/path/to/reference_assets/ref2va_image.png

curl -sS -X POST "${API_URL}" \
  -F 'prompt=A white cat sits on a beige couch and slowly looks toward the camera.' \
  -F 'aspect_ratio=adaptive' \
  -F 'short_edge=768' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3100' \
  -F 'extra_params={"task":"ref2va","duration":8.0,"audio_flow_shift":3.0}' \
  -F "input_reference=@${REF_IMAGE};type=image/png" \
  -o ref2va_image_only.mp4
```

An image-plus-audio request is:

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

### 4. Ref2VA: video, separate audio, and mixed references

Run this request against the combined service. Repeat the
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

The server stores uploaded references only for the lifetime of the request and
deletes temporary files after generation. A video may use its embedded
soundtrack, a separate `audio_reference`, or both. To send a mixed multipart
request, repeat `input_references` for each image, video, or audio file; the
server classifies them by MIME type and preserves the per-type order.

Reference videos must be MP4/MOV with H.264/H.265 video, optional AAC/MP3
audio, 2–15 seconds each, and at most 15 seconds combined. They may still be
longer than the generated clip. Use `start_time_seconds` to select a
synchronized segment; for multiple typed video references, pass one value per
video in `extra_params.start_time_seconds`.

Reference images accept JPG/JPEG, PNG, WEBP, HEIC, or HEIF up to 30 MiB. Standalone
audio references accept WAV or MP3 up to 15 MiB, with 2–15 seconds per file and
at most 15 seconds combined.

## Official input matrix and limits

| Task | Supported references | Limits |
|------|----------------------|--------|
| T2VA | text only | prompt must be non-empty |
| FL2VA | first image, last image, or ordered first+last images | at most 2 images; `frame_indices` is `[0]`, `[-1]`, or `[0,-1]` |
| Ref2VA | image-only, image+image, image+video, video+audio, and mixed image/video/audio | images ≤9, videos ≤3, audios ≤3, total references ≤12; audio requires a visual reference |

The H3 output contract is 4–15 seconds at 24 FPS, stereo 32 kHz audio, and a
32-pixel canvas multiple. T2VA requires one named output ratio from `21:9`,
`16:9`, `4:3`, `1:1`, `3:4`, or `9:16`. FL2VA always follows the first input
image's ratio and ignores a generic `aspect_ratio` override. Ref2VA defaults to
`16:9`; `adaptive` and SGLang's `auto` spelling are accepted aliases for that
default. `short_edge` controls the 768-pixel canvas and must be `768`.
`num_outputs_per_prompt` accepts 1–10 and derives each output seed as
`seed + output_index`. The asynchronous endpoint returns all
outputs; the synchronous raw-MP4 endpoint returns the first output when more
than one is requested.

## Key parameters

| Parameter | Recommended value | Notes |
|-----------|-------------------|-------|
| `task` | `t2va`, `fl2va`, or `ref2va` | Passed in `extra_params`; selects the task-specific DiT |
| `duration` | Workload-specific | Decimal seconds in `extra_params`; converted to H3-compatible frame count |
| `fps` | `24` | H3 output FPS is fixed |
| `num_inference_steps` | `50` | Matches the reference accuracy workloads |
| `flow_shift` | `12` | Video sigma shift |
| `audio_flow_shift` | `3` | Audio sigma shift, passed in `extra_params` |
| `seed` | Task-specific | Use a fixed value for reproducibility |
| `aspect_ratio` | Task-specific | T2VA requires a named ratio; FL2VA follows the input image; Ref2VA defaults to `16:9` |
| `short_edge` | `768` | H3 shape policy requires exactly 768 when `width`/`height` are omitted |
| `num_outputs_per_prompt` | `1` | 1–10; async API returns every output |
| `start_time_seconds` | `0` | Reference-video segment start; use a list in `extra_params` for multiple videos |
| `width`, `height` | Multiples of 32 | Output aspect ratio must be between 1:4 and 4:1 |

## ComfyUI Frontend

Users can also use a ComfyUI frontend to interact with a hosted MiniMax-H3 service. The ComfyUI frontend can run in a separate environment or machine. Refer to [vLLM-Omni ComfyUI Integration](../../docs/features/comfyui.md) for details.

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

## Validated FP8 evidence

With eager DiT/text-encoder TP2 and VAE tiling, the 384x672, 107-frame,
10-step quality case measured LPIPS 0.1156 (limit 0.20), PSNR 23.6316 dB,
audio spectral cosine 0.9589 (minimum 0.80), and audio RMS ratio 0.9342. The
resident per-GPU peak was 68.52 GiB for BF16 and 53.51 GiB for FP8, a 22%
reduction.

## Known limitations

- Combined serving requires sibling `FL2VA` and `Ref2VA` directories, loads
  both task-specific DiTs, and loads shared components once from `FL2VA`.
- H3 currently executes one generation request per diffusion batch.
- The first regional-compile request is a warmup and should not be included in
  steady-state performance measurements.
- Online FP8 is not compatible with layerwise offload.
- Image+audio Ref2VA accepts exactly one image and one audio reference.
- Video Ref2VA accepts one or more video files, but not an additional standalone
  audio reference.
- VAE patch parallelism requires size 1 or the full DiT group size and supports
  the H3 native `tile` mode only.

## Additional resources

- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
- [Diffusion parallelism](../../docs/user_guide/diffusion/parallelism/overview.md)
