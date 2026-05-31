# Text-To-Video

This example demonstrates how to deploy the Wan2.2 text-to-video model for online video generation using vLLM-Omni.

## Start Server

Wan2.2 T2V validation in this branch is intentionally scoped to the old
`feat/preemption-recovery-v0.16.0-squashed` serving topology:

- dispatcher on `:8080`
- single managed backend on `:8091`
- backend config fixed to:
  - `--usp 8`
  - `--enable-layerwise-offload`
  - `--boundary-ratio 0.875`
  - `--vae-use-slicing`
  - `--vae-use-tiling`

### Backend Start

Use the startup script for the backend:

```bash
bash run_server.sh
```

The script defaults to the supported validation config and allows overriding:
- `MODEL` (default: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`)
- `PORT` (default: `8091`)
- `USP` (default: `8`)
- `ENABLE_LAYERWISE_OFFLOAD` (default: `1`)
- `BOUNDARY_RATIO` (default: `0.875`)
- `FLOW_SHIFT` (default: `5.0`)
- `VAE_USE_SLICING` (default: `1`)
- `VAE_USE_TILING` (default: `1`)
- `CACHE_BACKEND` (default: `none`)
- `ENABLE_CACHE_DIT_SUMMARY` (default: `0`)

### Dispatcher Start

Run baseline or `super_p95` through the dispatcher even though there is only one
backend, so the serving topology stays identical to the old validation setup.

Baseline:

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
VLLM_OMNI_ENABLE_DIFFUSION_SERVER_SCHEDULING=0 \
VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=0 \
python3 benchmarks/diffusion/super_p95_dispatcher.py \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-url http://127.0.0.1:8091 \
  --hardware-profile 910B3
```

super_p95:

```bash
env NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
VLLM_OMNI_ENABLE_DIFFUSION_SERVER_SCHEDULING=1 \
VLLM_OMNI_ENABLE_DIFFUSION_PREEMPTION=1 \
python3 benchmarks/diffusion/super_p95_dispatcher.py \
  --host 0.0.0.0 \
  --port 8080 \
  --backend-url http://127.0.0.1:8091 \
  --hardware-profile 910B3
```

## Async Job Behavior

`POST /v1/videos` is asynchronous. It creates a video job and immediately
returns metadata like the job ID and initial `queued` status. To get the final
artifact, poll the job status and then download the completed file from the
content endpoint.

The main endpoints are:
- `POST /v1/videos`: create a video generation job
- `GET /v1/videos/{video_id}`: retrieve the current job status and metadata
- `GET /v1/videos`: list stored video jobs
- `GET /v1/videos/{video_id}/content`: download the generated video file
- `DELETE /v1/videos/{video_id}`: delete the job and any stored output

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
create_response=$(curl -s http://localhost:8091/v1/videos \
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
  status=$(curl -s "http://localhost:8091/v1/videos/${video_id}" | jq -r '.status')
  if [ "$status" = "completed" ]; then
    break
  fi
  if [ "$status" = "failed" ]; then
    echo "Video generation failed"
    exit 1
  fi
  sleep 2
done

curl -s "http://localhost:8091/v1/videos/${video_id}" | jq .
curl -L "http://localhost:8091/v1/videos/${video_id}/content" -o wan22_output.mp4
```

## Request Format

### Simple Text-to-Video Generation

```bash
curl -X POST http://localhost:8091/v1/videos \
  -F "prompt=A cinematic view of a futuristic city at sunset"
```

### Generation with Parameters

```bash
curl -X POST http://localhost:8091/v1/videos \
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
curl -s http://localhost:8091/v1/videos/${video_id} | jq .
```

### List jobs

```bash
curl -s http://localhost:8091/v1/videos | jq .
```

### Download the completed video

```bash
curl -L http://localhost:8091/v1/videos/${video_id}/content -o wan22_output.mp4
```

### Delete a job and its stored file

```bash
curl -X DELETE http://localhost:8091/v1/videos/${video_id} | jq .
```

## Poll Until Complete

```bash
while true; do
  status=$(curl -s http://localhost:8091/v1/videos/${video_id} | jq -r '.status')
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
