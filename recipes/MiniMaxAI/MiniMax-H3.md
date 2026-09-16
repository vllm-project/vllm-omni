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
- `Ref2VA`: mixed image/video/audio conditioning (`ref2va`); supported
  combinations and counts are listed in the input matrix below

One vLLM-Omni diffusion stage can load both DiTs while instantiating the
tokenizer, processor, Qwen3-VL text encoder, video VAE, and audio VAE only
once. Requests select the DiT with `extra_params.task`.

The generated MP4 contains H.264 video and synchronized stereo audio.

## Official input matrix and limits

| Task | Supported references | Limits |
| ------ | ---------------------- | -------- |
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

## Choose a deployment

| Deployment | Guide |
| --- | --- |
| NVIDIA CUDA | [Deployment guide](MiniMax-H3-CUDA.md) |
| RTX 4090 | [Hardware recipe](MiniMax-H3-4090.md) |
| RTX 5090 | [Hardware recipe](MiniMax-H3-5090.md) |
| RTX PRO 5000 | [Hardware recipe](MiniMax-H3-RTX-PRO-5000.md) |
| RTX PRO 6000 | [Hardware recipe](MiniMax-H3-RTX-PRO-6000.md) |
| DGX Spark (GB10) | [Hardware recipe](MiniMax-H3-Spark-GB10.md) |
| AMD Instinct | <a id="amd-rocm"></a>[ROCm deployment](MiniMax-H3-ROCm.md) |
| Ascend NPU | [Atlas A3](MiniMax-H3-NPU.md), [950PR](MiniMax-H3-NPU-950PR.md) |
| Moore Threads MUSA | [Hardware recipe](MiniMax-H3-MUSA.md) |
| Separate text encoder stage | [Disaggregated deployment](MiniMax-H3-Disaggregated.md) |
| FastH3 | [VSA serving](#fasth3-vsa-serving), [Dense](#fasth3-dense) |

Follow the selected deployment's setup, then use the [HTTP API examples](#http-api-examples).
For optional behavior, see [Optimization options](#optimization-options),
[LoRA](#lora), and the validation results in each deployment guide.

## Prerequisites

The checkpoint requires Hugging Face access approval. Authenticate once;
`vllm serve` downloads the required components automatically:

```bash
hf auth login
```

The vLLM-Omni pipeline downloads `FL2VA/**`, `Ref2VA/model_index.json`, and
`Ref2VA/transformer/**`. It does not download or load the diffusers-format
`transformer`, `transformer_ref`, or `vae` weights at the repository root, nor
duplicate Ref2VA copies of shared components.

Install vLLM-Omni using the [installation guide](../../docs/getting_started/installation/README.md)
for the target platform.

`ffmpeg` and `ffprobe` must be available on `PATH`. They are used for
reference-video preparation and MP4 output.

## Start a server

Use a command from the [deployment guide](#choose-a-deployment) for the target
hardware. The repository ID loads a combined service; `--task-type fl2va` or
`--task-type ref2va` selects one task partition. T2VA uses the FL2VA partition.

The pipeline uses `FL2VA` for model discovery and shared components, and loads
the second DiT from `Ref2VA/transformer` for combined serving.

### Checkpoint storage

Each H3 checkpoint partition (`FL2VA` or `Ref2VA`) contains about **134 GiB** of
BF16 safetensors (about **135 GiB** on disk). Keeping both partitions
locally therefore needs roughly **270 GiB** of model storage. A combined
service downloads both; `--task-type fl2va` or `--task-type ref2va` downloads
only the selected partition.

## HTTP API examples

The following requests use the synchronous endpoint so the returned body can
be saved directly as an MP4. The asynchronous `POST /v1/videos` endpoint can
also be used when job polling is preferred.

All four tasks use 24 FPS, 50 sigma points, seed values from the validated
workloads, and the checkpoint-reference video/audio flow shifts of 12 and 3.
Decimal durations are passed through `extra_params`.

### 1. T2VA: text to video and audio

Run this request against the combined service:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/videos/sync" \
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
curl -sS -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=A man stands beside a yellow car at night. The car drives away; he follows it with his eyes and begins singing sadly, with synchronized voice and city ambience.' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=2101' \
  -F 'extra_params={"task":"fl2va","duration":8.7,"audio_flow_shift":3.0}' \
  -F "input_reference=@/path/to/fl2va_first_frame.png;type=image/png" \
  -o fl2va.mp4
```

Use the same `FL2VA` partition for the official tail-keyframe forms. A single
image with `frame_indices=[-1]` conditions the last frame; two ordered images
with `frame_indices=[0,-1]` condition the first and last frames:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=The subject moves naturally from the first image to the last image.' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=2102' \
  -F 'extra_params={"task":"fl2va","duration":8.7,"frame_indices":[0,-1],"audio_flow_shift":3.0}' \
  -F "input_references=@/path/to/fl2va_first_frame.png;type=image/png" \
  -F "input_references=@/path/to/fl2va_last_frame.png;type=image/png" \
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
curl -sS -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=A white cat sits on a beige couch and slowly looks toward the camera.' \
  -F 'aspect_ratio=adaptive' \
  -F 'short_edge=768' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3100' \
  -F 'extra_params={"task":"ref2va","duration":8.0,"audio_flow_shift":3.0}' \
  -F "input_reference=@/path/to/reference_assets/ref2va_image.png;type=image/png" \
  -o ref2va_image_only.mp4
```

An image-plus-audio request is:

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=A white cat with black mustache and eyebrow markings sits on a beige couch, lip-syncing precisely to the complete reference audio before shifting from confusion to deadpan speechlessness.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3101' \
  -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
  -F "input_reference=@/path/to/reference_assets/ref2va_image.png;type=image/png" \
  -F "audio_reference={\"audio_url\":\"http://127.0.0.1:8092/ref2va_audio.mp3\"}" \
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
curl -sS -X POST "http://127.0.0.1:8000/v1/videos/sync" \
  -F 'prompt=Remove the green screen background of Video 1 and replace it with the fairytale environment from Video 2. Match the background motion to the character actions and relight the character to fit the scene.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12' \
  -F 'seed=3101' \
  -F 'extra_params={"task":"ref2va","duration":15.0,"audio_flow_shift":3.0}' \
  -F "input_references=@/path/to/green_screen_subject.mp4;type=video/mp4" \
  -F "input_references=@/path/to/fairytale_background.mov;type=video/quicktime" \
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

### Key parameters

| Parameter | Recommended value | Notes |
| ----------- | ------------------- | ------- |
| `quality` | omitted or `lossless` | Request-level quality intent; `high` dynamically installs H3's conservative Cache-DiT profile |
| `extra_params.force_refresh_step_hint` | omitted | Optional positive 1-based denoising-step hint for an active Cache-DiT request; pair with `extra_params.force_refresh_step_policy`=`once` or `repeat` |
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

### ComfyUI Frontend

Users can also use a ComfyUI frontend to interact with a hosted MiniMax-H3 service. The ComfyUI frontend can run in a separate environment or machine. Refer to [vLLM-Omni ComfyUI Integration](../../docs/features/comfyui.md) for details.

## FastH3 adapter

[FastH3](https://haoailab.com/blogs/fasth3-preview/) is FastVideo's four-step
DMD2 student of H3-Base. It generates video with synchronized audio using the
base H3 checkpoint and a variant-specific adapter.

| Setting | Supported configuration |
| --- | --- |
| Task | Text-to-video-and-audio (`t2va`) only |
| Inference steps | 4 |
| VSA parallelism | Local attention or pure Ulysses; ring/all-gather SP unsupported |
| Weight loading | Adapter fused at startup; offload and per-request LoRA unsupported |

### FastH3 VSA serving

Follow the [FastH3 VSA deployment example](MiniMax-H3-CUDA.md#fasth3-vsa)
for installation, adapter download, startup, and a T2VA request. The options
below define the model contract for that deployment.

The output is an MP4 containing video and audio. Startup logs include
`FastH3 adapter active`; DiT execution logs `FASTVIDEO_VSA H3 routing`.
With this global backend selection, the token refiner uses dense SDPA;
its missing-grid fallback warning is expected.

| Option | Meaning |
| --- | --- |
| `--lora-path` | Selects and loads the adapter at startup; selecting VSA alone does not load it |
| `--task-type fl2va` | Loads the partition that serves T2VA, avoiding the additional Ref2VA model loaded by default from the repository root |
| `--usp N` | Ulysses worker count; choose a count that fits the resident model weights and activations |
| `--fastvideo-vsa-topk K` | Video blocks retained per query; default 64, with all prefix blocks retained |

Smaller top-k values may reduce both attention cost and output quality. Keep
the checkpoint's default video/audio flow shifts and the four-step schedule.

### FastH3 Dense

To use Dense / Data-Free, replace `vsa-datafree` with `dense-datafree` in the
download and serve commands in the [deployment example](MiniMax-H3-CUDA.md#fasth3-vsa),
and omit `--diffusion-attention-backend FASTVIDEO_VSA`
to use the platform's dense default. This variant does not need `fastvideo-kernel`.

Deployment-specific measurements are recorded in the
[FastH3 Dense validation](MiniMax-H3-CUDA.md#fasth3-dense-validation).

## Optimization options

### Attention Backends

The platform selects the default dense attention backend. Backend availability,
installation, and hardware-specific tuning are described in the deployment
guides. See the [attention backend guide](../../docs/user_guide/diffusion/attention_backends.md)
for the shared configuration interface, and [FastH3 VSA](#fasth3-vsa-serving)
for the adapter-specific path.

Quantized attention and Skip-Softmax trade fidelity for speed. Compare against
dense output with the same prompt and seed before adopting them; the
[CUDA recipe](MiniMax-H3-CUDA.md#attention-backends) records the supported settings.

### Text encoder tensor parallelism

`--text-encoder-tp-size N` shards the retained Qwen3-VL text decoder across the
first `N` DiT ranks. `N` must divide its 64 attention heads and 8 KV heads and
must not exceed the DiT worker count. Encoder ranks use their own process group;
row-parallel projections are all-reduced, and the full `[seq, 5120]` layer-50
hidden state is replicated on each rank, matching the reference path within
BF16 rounding. See the deployment guides for memory budgets and launch examples.

### Step execution and continuous batching

H3 implements the step-wise execution contract, so the scheduler can admit and
retire requests between denoise steps instead of running one request end to
end. Add the feature gate, then raise `--max-num-seqs` to co-batch:

```bash
--step-execution --max-num-seqs 4
```

Request mode executes one generation request per diffusion batch.
Co-batched requests in step mode are packed into a single sequence that keeps
one attention document per request, so a batch costs one DiT forward. Packing requires
`--diffusion-attention-backend FLASH_ATTN`; other backends fall back to one
forward per request. `--max-num-seqs 1` keeps the conservative single-request
step path. Cache acceleration (`--cache-backend`) is not available in step mode.

Co-batching is not a throughput guarantee. See the
[measured CUDA workloads](MiniMax-H3-CUDA.md#step-execution-measurements)
for the observed trade-offs and unmeasured arrival patterns.

### Online FP8 quantization

MiniMax H3 supports online FP8 quantization of both the DiT and the Qwen3-VL
text decoder. The checkpoint remains BF16 on disk; vLLM-Omni creates FP8
weights at runtime and uses dynamic activation scaling during inference. By
default, `--quantization fp8` quantizes eligible attention and MLP linears in
the text decoder, token refiner, and main DiT blocks, as well as the DiT
condition and AdaLN projections. The Qwen vision tower, embeddings, norms,
RoPE, both VAEs, and the model's FP32 patch, timestep, and output projections
keep checkpoint precision.

To select a component, use `--diffusion-quantization-config` with
`{"transformer":{"method":"fp8"}}` for DiT-only FP8 or
`{"text_encoder":{"method":"fp8"}}` for text-decoder-only FP8. The two
entries can be combined. The shorthand below enables both components.

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

The structured option replaces `--quantization fp8`. Online FP8 can be used
with H3 layerwise offload and with either DiT DLO transfer. DiT `allgather`
uses the ordinary loader to finalize FP8 weights and scales before sharding
them across ranks. DiT `rank-local` instead retains complete loader-produced
tensors and avoids the synchronized request-wave contract. H3's TP-sharded
text encoder uses `rank-local`.

### Request-scoped quality

Add one of these fields to any HTTP request above. No Cache-DiT startup option
is required; H3 installs its conservative profile when a `high` request
arrives and removes it for a `lossless` request.

```bash
-F 'quality=lossless'  # Native reference path for this request.
-F 'quality=high'      # H3's conservative Cache-DiT profile.
```

Omitting `quality` preserves the startup default: it uses the reference path
normally, or the server-configured profile when the server was started with
`--cache-backend cache_dit`.

Turbo is independent of this quality switch: `quality` selects a Cache-DiT
policy, while Turbo changes the active LoRA weights and sampling schedule. See
[LoRA](#lora) below.

The [CUDA validation](MiniMax-H3-CUDA.md#request-scoped-quality-validation)
records measured latency and fidelity for this profile. Results depend on
hardware, topology, and workload; `lossless` remains the exact reference path.

### TeaCache acceleration

TeaCache reuses DiT block residuals across denoising steps when consecutive
timestep embeddings are similar. MiniMax-H3 TeaCache is currently calibrated
only for the FL2VA partition. In combined serving, FL2VA requests use TeaCache
while Ref2VA requests run uncached; Ref2VA-only serving rejects TeaCache.

TeaCache and Cache-DiT are mutually exclusive; pick one cache backend per server.

The model-specific default and examples use `rel_l1_thresh=0.17`, which
provided the best conservative speed/quality balance in the validated 107-frame
T2VA workload. Lower values
may produce few or no cache hits, while higher values can improve performance
at the cost of output quality. Validate the threshold on representative
prompts and generation settings before changing it.

#### Offline (Python API)

Set `model` to the local checkpoint partition:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="/path/to/MiniMax-H3/FL2VA",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.17},
    trust_remote_code=True,
    enable_cpu_offload=True,
)
outputs = omni.generate(
    "A quiet cinematic night scene with matching ambient sound.",
    OmniDiffusionSamplingParams(
        height=256,
        width=448,
        num_frames=29,
        fps=24,
        num_inference_steps=50,
        seed=42,
        extra_args={
            "task": "t2va",
            "duration": 4.0,
            "aspect_ratio": "16:9",
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
        },
    ),
)
```

#### Online serving

```bash
vllm serve /path/to/MiniMax-H3/FL2VA \
  --omni \
  --trust-remote-code \
  --cache-backend tea_cache \
  --cache-config '{"rel_l1_thresh":0.17}'
```

## LoRA

Turbo and FlashGen use runtime adapters with artifact-specific sampling
contracts. A checkpoint that instead pins `base_schedule` in `model_index.json`
uses the interval count: for example, four steps for
`[1.0, 0.7, 0.4, 0.15, 0.0]`. Follow the contract for the selected artifact;
the same numerical step argument does not have identical meaning across families.

### Turbo LoRA

The eight Diffusers-layout LightX2V Turbo artifacts are supported. The
filename records the contract, so the server reads the step count, task family
and flow shift from it and validates each request against the artifact that is
loaded. It does not rewrite request or deploy-config sampling values: the
request must carry that artifact's own settings, listed here, or it is
rejected.

| Artifact | Task | Forwards | `num_inference_steps` | `flow_shift` | declared `alpha` |
| --- | --- | ---: | ---: | ---: | ---: |
| `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | T2VA / FL2VA | 4 | 5 | 12 | none -> 8 |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` | T2VA / FL2VA | 4 | 5 | 6 | 128 |
| `minimax_h3_fl2v_turbo_4step_v1.1_768p_bf16.safetensors` | T2VA / FL2VA | 4 | 5 | 6 | 128 |
| `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` | T2VA / FL2VA | 4 | 5 | 6 | 8 |
| `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` | T2VA / FL2VA | 8 | 9 | 12 | 8 |
| `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` | T2VA / FL2VA | 8 | 9 | 6 | 8 |
| `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` | Ref2VA | 4 | 5 | 12 | 8 |
| `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` | Ref2VA | 8 | 9 | 6 | 8 |

`audio_flow_shift` is `3.0` across the family. Each row is the complete
published filename; preserve it when downloading an artifact.

Alpha needs no manual compensation: the server reads it from the artifact's
metadata, falling back to 8 with a warning for
`minimax_h3_fl2v_turbo_4step_v0.1`, the one artifact that declares none. The
request-level `scale` is a further multiplier on top of it.

> [!NOTE]
> Every artifact is rank 128 and the delta is applied at `scale * alpha /
> rank`, so the `alpha=8` rows drive at 1/16 the strength of the `alpha=128`
> rows. 8 is the default of LightX2V's reference script, which never reads the
> metadata; its documented `v0.1` command does not override that default.

Every artifact except `minimax_h3_fl2v_turbo_4step_v0.1` also ships a
`_comfyui_` export of the same weights. Those fuse Q/K/V into one projection
and are **not** supported; downloading one is refused by name. Take the
Diffusers file. The filename is the contract, so do not rename an artifact
either -- a renamed file is rejected rather than served on a guess.

FL2VA artifacts serve `t2va` and `fl2va` on any FL2VA or combined server.
Ref2VA artifacts require
`--task-type ref2va`: a combined server serves `ref2va` from a second DiT that
the adapter cannot bind to, so loading one there is refused rather than silently
running an undistilled model on the few-step schedule.

Download the artifact you want:

```bash
hf download lightx2v/Minimax-h3-Turbo "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors" --local-dir "/path/to/minimax-h3-turbo"
```

`--lora-path` accepts one artifact, or a directory holding exactly one.

A directory containing several recognized artifacts is ambiguous and rejected;
pass the intended artifact's filename explicitly.

Start from a non-offloaded or DLO FL2VA server command and add
`--task-type fl2va --lora-backend peft --lora-path "/path/to/minimax-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"`.
`--lora-path` preloads the adapter; each request still activates it and must
carry that artifact's sampling settings:

```bash
-F 'num_inference_steps=5' \
-F 'flow_shift=6' \
-F 'extra_params={"task":"t2va","duration":4.4,"audio_flow_shift":3.0}' \
-F "lora={\"name\":\"h3-turbo-v1.0\",\"path\":\"/path/to/minimax-h3-turbo/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors\",\"scale\":1.0}"
```

Switching to another FL2VA artifact requires updating both `--lora-path` and
the request's `lora.path`, and carrying that row's
`num_inference_steps` and `flow_shift`: `9` and `6` for `8step_v1.0_768p`, `9`
and `12` for the 544p `8step_v1.0`. A request that does not match the loaded
artifact is rejected, so a mismatch cannot silently degrade output.

The two `ref2v` rows are not served by this FL2VA command. Start a
`--task-type ref2va` server, take a request from
[Ref2VA](#3-ref2va-image-only-imageaudio-or-mixed-references) and override
`num_inference_steps` and `flow_shift` with that row's values; those examples
already send `audio_flow_shift=3.0`.

For FL2VA, change `task` and add `input_reference` as shown above. Turbo is
dynamic-only and does not support prefusion or LoRA composition. The requested
sigma points always number one more than the artifact's denoiser evaluations.
See [runtime adapter residency](#runtime-adapter-residency) for offload support.

### FlashGen native LoRA

The FlashGen 4-step T2VA artifact uses the native MiniMax-H3 module layout and
declares its distilled sigma schedule in safetensors metadata. It is published on
[ModelScope](https://modelscope.cn/models/FlashGen/Minimax-H3-4step-lora-flashgen):

```text
FlashGen/Minimax-H3-4step-lora-flashgen/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors
```

Download only that file:

```bash
python -m pip install modelscope
modelscope download FlashGen/Minimax-H3-4step-lora-flashgen \
  --local_dir "/path/to/minimax-h3-flashgen-lora" \
  --include "minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"
```

Start from a non-offloaded or DLO FL2VA server command and add
`--task-type fl2va --lora-backend peft --lora-path "/path/to/minimax-h3-flashgen-lora/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors"`.
Each request must use T2VA and the distilled interval-count contract:

```bash
-F 'num_inference_steps=4' \
-F 'extra_params={"task":"t2va","duration":5.2}' \
-F "lora={\"name\":\"h3-flashgen-v1.0\",\"path\":\"/path/to/minimax-h3-flashgen-lora/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors\",\"scale\":1.0}"
```

This path rejects Ref2VA and checkpoints that already pin `base_schedule` in
`model_index.json`. The adapter metadata carries
`base_schedule=1.0,0.7,0.4,0.15,0.0`, so `num_inference_steps=4` means four
denoiser evaluations, not five sigma points. Request-mode generation may omit
the field and take the count from the adapter schedule; `--step-execution`
requires it explicitly, because the step scheduler reads the total step count
off the request at admission, before the adapter schedule is known.

See [runtime adapter residency](#runtime-adapter-residency) for offload and
memory requirements.

To validate a deployment, post the same fixed-seed T2VA request twice with the
adapter and twice without it, then compare the four output digests. The adapter
is bound and deterministic when each pair matches internally and the two pairs
differ from each other.

### Runtime adapter residency

Turbo and FlashGen support DLO in request mode. Their LoRA tensors remain
resident while base blocks are streamed, so budget additional accelerator
memory. Model-level and standard layerwise offload are unsupported for these
runtime adapters, and step execution cannot be combined with DLO.

Pure Ulysses replicates the adapter on each rank; DiT tensor parallelism shards
its output projections. Measure the resident footprint for the selected
parallel layout rather than estimating it from the adapter file size.

## Known limitations

Input constraints are defined in the [input matrix](#official-input-matrix-and-limits).
Cache and batching restrictions are described with their
[optimization options](#optimization-options).

- `--cfg-parallel-size > 1` is rejected by design (CFG-distilled, no negative branch).
- VAE patch parallelism requires size 1 or the full DiT group size and supports the
  H3 native `tile` mode only.
- Ulysses x Ring hybrid attention supports H3's single-request contiguous suffix
  padding. Arbitrary attention masks and multi-request packed batches remain
  unsupported on the Ring path.
- Online FP8 with DLO AllGather temporarily materializes the complete FP8 model
  in host memory on every rank during startup before retaining only each rank's
  shard. Size startup host memory for that transient peak.
- Pure Ulysses still replicates the full DiT on every rank, so smaller-memory GPUs
  cannot use `--usp N --tp 1` as a resident capacity path. Use DiT tensor parallelism
  or model-level CPU offload; text-encoder TP alone is not sufficient.

## Additional resources

- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
- [Diffusion parallelism](../../docs/user_guide/diffusion/parallelism/overview.md)
