# Text-To-Video

This example demonstrates how to deploy text-to-video models for online video generation using vLLM-Omni.

## Supported Models

| Model | Model ID |
|-------|----------|
| Wan2.1 T2V (1.3B) | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| Wan2.1 T2V (14B) | `Wan-AI/Wan2.1-T2V-14B-Diffusers` |
| Wan2.2 T2V | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` |
| LTX-2 | `Lightricks/LTX-2` |

## Wan2.2 T2V

### Start Server

#### Basic Start

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8000
```

#### Start with Parameters

Or use the startup script:

```bash
bash run_server.sh
```

The script allows overriding:
- `MODEL` (default: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`)
- `PORT` (default: `8000`)
- `BOUNDARY_RATIO` (default: `0.875`)
- `FLOW_SHIFT` (default: `5.0`)
- `CACHE_BACKEND` (default: `none`)
- `ENABLE_CACHE_DIT_SUMMARY` (default: `0`)

## Async Job Behavior

`POST /v1/videos` is asynchronous. It creates a video job and immediately
returns metadata like the job ID and initial `queued` status. To get the final
artifact, poll the job status and then download the completed file from the
content endpoint.

The main endpoints are:
- `POST /v1/videos`: create a video generation job (async)
- `POST /v1/videos/sync`: generate a video and return raw bytes (sync, for benchmarks)
- `GET /v1/videos/{video_id}`: retrieve the current job status and metadata
- `GET /v1/videos`: list stored video jobs
- `GET /v1/videos/{video_id}/content`: download the generated video file
- `DELETE /v1/videos/{video_id}`: delete the job and any stored output

## Sync API (Benchmark / Testing)

`POST /v1/videos/sync` is a synchronous alternative that blocks until generation
completes and returns the raw video bytes (`video/mp4`) directly in the response
body. It is designed for benchmark and testing scenarios where one-shot
request/response latency measurement is needed.

The sync endpoint accepts the same form parameters as `POST /v1/videos`. It does
not create any stored job record — the response is purely the generated video
file. Metadata is returned via response headers:

- `X-Request-Id`: unique identifier for this generation request
- `X-Model`: model name used for generation
- `X-Inference-Time-S`: wall-clock inference time in seconds

```bash
curl -X POST http://localhost:8000/v1/videos/sync \
  -F "prompt=Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  -F "size=832x480" \
  -F "num_frames=33" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -F "guidance_scale_2=4.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=5.0" \
  -F "seed=42" \
  -o sync_t2v_output.mp4
```

## Storage

Generated video files are stored on local disk by the async video API.
Local file storage behavior can be controlled via the following environment variables:

- `VLLM_OMNI_STORAGE_PATH`: directory used for generated files (default: `/tmp/storage`)
- `VLLM_OMNI_STORAGE_MAX_CONCURRENCY`: max concurrent save/delete operations (default: `4`)

Example:

```bash
export VLLM_OMNI_STORAGE_PATH=/var/tmp/vllm-omni-videos
export VLLM_OMNI_STORAGE_MAX_CONCURRENCY=8
```

## API Calls

### Method 1: Using curl

```bash
# Basic text-to-video generation
bash run_curl_text_to_video.sh

# Or execute directly (OpenAI-style multipart)
create_response=$(curl -s http://localhost:8000/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=33" \
  -F "negative_prompt=色调艳丽 ，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -F "guidance_scale_2=4.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=5.0" \
  -F "seed=42")

video_id=$(echo "$create_response" | jq -r '.id')
while true; do
  status=$(curl -s "http://localhost:8000/v1/videos/${video_id}" | jq -r '.status')
  if [ "$status" = "completed" ]; then
    break
  fi
  if [ "$status" = "failed" ]; then
    echo "Video generation failed"
    exit 1
  fi
  sleep 2
done

curl -s "http://localhost:8000/v1/videos/${video_id}" | jq .
curl -L "http://localhost:8000/v1/videos/${video_id}/content" -o wan22_output.mp4
```

## Request Format

### Simple Text-to-Video Generation

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cinematic view of a futuristic city at sunset"
```

### Generation with Parameters

```bash
curl -X POST http://localhost:8000/v1/videos \
  -F "prompt=A cinematic view of a futuristic city at sunset" \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=33" \
  -F "negative_prompt=low quality, blurry, static" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -F "guidance_scale_2=4.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=5.0" \
  -F "seed=42"
```

## Generation Parameters

| Parameter             | Type   | Default | Description                                      |
| --------------------- | ------ | ------- | ------------------------------------------------ |
| `prompt`              | str    | -       | Text description of the desired video            |
| `seconds`             | str    | None    | Clip duration in seconds                         |
| `size`                | str    | None    | Output size in `WIDTHxHEIGHT` format             |
| `negative_prompt`     | str    | None    | Negative prompt                                  |
| `width`               | int    | None    | Video width in pixels                            |
| `height`              | int    | None    | Video height in pixels                           |
| `num_frames`          | int    | None    | Number of frames to generate                     |
| `fps`                 | int    | None    | Frames per second for output video               |
| `num_inference_steps` | int    | None    | Number of denoising steps                        |
| `guidance_scale`      | float  | None    | CFG guidance scale (low-noise stage)             |
| `guidance_scale_2`    | float  | None    | CFG guidance scale (high-noise stage, Wan2.2)     |
| `boundary_ratio`      | float  | None    | Boundary split ratio for low/high DiT (Wan2.2)   |
| `flow_shift`          | float  | None    | Scheduler flow shift (Wan2.2)                    |
| `seed`                | int    | None    | Random seed (reproducible)                       |
| `lora`                | object | None    | LoRA configuration                               |

## Create Response Format

`POST /v1/videos` returns a job record, not inline base64 video data.

```json
{
  "id": "video_gen_123",
  "object": "video",
  "status": "queued",
  "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  "prompt": "A cinematic view of a futuristic city at sunset",
  "created_at": 1234567890
}
```

## Retrieve, List, Download, and Delete

### Retrieve a job

```bash
curl -s http://localhost:8000/v1/videos/${video_id} | jq .
```

### List jobs

```bash
curl -s http://localhost:8000/v1/videos | jq .
```

### Download the completed video

```bash
curl -L http://localhost:8000/v1/videos/${video_id}/content -o wan22_output.mp4
```

### Delete a job and its stored file

```bash
curl -X DELETE http://localhost:8000/v1/videos/${video_id} | jq .
```

## Poll Until Complete

```bash
while true; do
  status=$(curl -s http://localhost:8000/v1/videos/${video_id} | jq -r '.status')
  if [ "$status" = "completed" ]; then
    break
  fi
  if [ "$status" = "failed" ]; then
    echo "Video generation failed"
    exit 1
  fi
  sleep 2
done
```

## LTX-2

### Start Server

#### Basic Start

```bash
vllm serve Lightricks/LTX-2 --omni --port 8098 \
    --enforce-eager --flow-shift 1.0 --boundary-ratio 1.0
```

#### Start with Optimization Presets

Use the LTX-2 startup script with built-in optimization presets:

```bash
# Baseline (1 GPU, eager)
bash run_server_ltx2.sh baseline

# 4-GPU Ulysses sequence parallelism (lossless)
bash run_server_ltx2.sh ulysses4

# Cache-DiT lossy acceleration (1 GPU, ~1.4× speedup)
bash run_server_ltx2.sh cache-dit

# Best combo: 4-GPU Ulysses SP + Cache-DiT (~2.2× speedup)
bash run_server_ltx2.sh best-combo
```

#### Optimization Benchmarks

Benchmarked on H800, online serving (480×768, 41 frames, 20 steps, `seed=42`).
"Inference" is the server-reported inference time; excludes HTTP/poll overhead.

| Preset | Server Command | Inference (s) | Speedup | Type |
|--------|---------------|---------------|---------|------|
| `baseline` | `--enforce-eager` | 10.3 | 1.00× | — |
| `compile` | *(default, no --enforce-eager)* | ~10.3 (warm) | ~1.00× | Lossless |
| `ulysses4` | `--enforce-eager --usp 4` | ~10.3 | ~1.00× | Lossless |
| `cache-dit` | `--enforce-eager --cache-backend cache_dit` | 7.4 avg | ~1.4× | Lossy |
| `best-combo` | `--enforce-eager --usp 4 --cache-backend cache_dit` | 4.7 avg | **~2.2×** | Lossless + Lossy |

**Observations**:
- **torch.compile**: On H800, warm-request inference time matches the eager baseline (~10.3s).
  The first request pays ~6s compilation overhead. Benefit depends on model architecture and GPU.
- **Ulysses SP (4 GPU)**: No measurable speedup alone for 41-frame generation at this resolution.
  Communication overhead outweighs gains at this sequence length.
- **Cache-DiT**: Inference varies per request (6–10s) due to dynamic caching decisions.
  Average is ~7.4s (~1.4× speedup) with slight quality tradeoff.
- **Best combo**: 4-GPU Ulysses SP + Cache-DiT synergize well — Cache-DiT reduces per-step
  computation, making the communication overhead of Ulysses SP worthwhile. Average ~4.7s
  (~2.2× speedup).
- **FP8 quantization**: Reduces VRAM but does not speed up LTX-2 on H800 (compute-bound).

**Deployment Recommendations**:
- For **production with quality priority**: use `baseline` with `--enforce-eager`
- For **maximum throughput** (4 GPUs, quality tradeoff): use `best-combo` (~2.2× speedup)
- For **single-GPU throughput**: use `cache-dit` (~1.4× speedup)
- `--enforce-eager` is recommended to avoid torch.compile warmup latency on first request

### Send Requests (curl)

```bash
# Using the provided script
bash run_curl_ltx2.sh

# Or directly
curl -sS -X POST http://localhost:8098/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=A serene lakeside sunrise with mist over the water." \
  -F "width=768" \
  -F "height=480" \
  -F "num_frames=41" \
  -F "fps=24" \
  -F "num_inference_steps=20" \
  -F "guidance_scale=3.0" \
  -F "seed=42"
```
