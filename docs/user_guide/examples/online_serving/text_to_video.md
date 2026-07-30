# Text-To-Video

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/text_to_video>.


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
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8091
```

#### Start with Parameters

Or use the startup script:

```bash
bash run_server.sh
```

The script allows overriding:
- `MODEL` (default: `Wan-AI/Wan2.2-T2V-A14B-Diffusers`)
- `PORT` (default: `8091`)
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
curl -X POST http://localhost:8091/v1/videos/sync \
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

- `VLLM_OMNI_SERVER_STORAGE__PATH`: directory used for generated files
  (default: `/tmp/storage`)
- `VLLM_OMNI_SERVER_STORAGE__FILE_CONCURRENCY`: max concurrent save/delete/open
  operations (default: `4`)
- `VLLM_OMNI_SERVER_STORAGE__FILE_TTL`: optional TTL for generated files in
  seconds
- `VLLM_OMNI_SERVER_STORAGE__TTL_SWEEP_INTERVAL`: optional sweep interval in
  seconds for enforcing file TTL; defaults to `300` when TTL is set

Example:

```bash
export VLLM_OMNI_SERVER_STORAGE__PATH=/var/tmp/vllm-omni-videos
export VLLM_OMNI_SERVER_STORAGE__FILE_CONCURRENCY=8
export VLLM_OMNI_SERVER_STORAGE__FILE_TTL=86400
export VLLM_OMNI_SERVER_STORAGE__TTL_SWEEP_INTERVAL=300
```

See also: [Configuration Options](../../../configuration/README.md)

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
  -F "enable_frame_interpolation=true" \
  -F "frame_interpolation_exp=1" \
  -F "frame_interpolation_scale=1.0" \
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
| `num_frames`          | int    | 1       | Number of frames to generate; set explicitly for video generation |
| `fps`                 | int    | None    | Frames per second for output video               |
| `num_inference_steps` | int    | None    | Number of denoising steps                        |
| `guidance_scale`      | float  | None    | CFG guidance scale (low-noise stage)             |
| `guidance_scale_2`    | float  | None    | CFG guidance scale (high-noise stage, Wan2.2)     |
| `boundary_ratio`      | float  | None    | Boundary split ratio for low/high DiT (Wan2.2)   |
| `flow_shift`          | float  | None    | Scheduler flow shift (Wan2.2)                    |
| `seed`                | int    | None    | Random seed (reproducible)                       |
| `lora`                | object | None    | LoRA configuration                               |
| `enable_frame_interpolation` | bool | false | Enable RIFE frame interpolation before MP4 encoding |
| `frame_interpolation_exp` | int | 1 | Interpolation exponent; 1=2x temporal resolution, 2=4x |
| `frame_interpolation_scale` | float | 1.0 | RIFE inference scale; use 0.5 for high-resolution inputs |
| `frame_interpolation_model_path` | str | None | Local directory or Hugging Face repo ID with `flownet.pkl`; defaults to `elfgum/RIFE-4.22.lite` |

## Frame Interpolation

Frame interpolation is an optional post-processing step for `/v1/videos` and
`/v1/videos/sync`. It synthesizes intermediate frames between generated frames
without rerunning the diffusion model. If the generated video has `N` frames,
the interpolated output frame count is `(N - 1) * 2**exp + 1`. The encoder FPS
is multiplied by `2**exp` so the output duration remains close to the original.

Frame interpolation runs in the diffusion worker post-processing path instead of
the API server encoding path, so it can reuse the worker's current accelerator
device without blocking the FastAPI event loop.

Example: generate 5 frames and interpolate to 9 frames:

```bash
curl -X POST http://localhost:8091/v1/videos/sync \
  -F "prompt=A dog running through a park" \
  -F "num_frames=5" \
  -F "fps=8" \
  -F "enable_frame_interpolation=true" \
  -F "frame_interpolation_exp=1" \
  -F "frame_interpolation_scale=1.0" \
  -o sync_t2v_interpolated.mp4
```

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

## LTX-2

```bash
vllm serve Lightricks/LTX-2 --omni --port 8098
```

See the [LTX-2 recipe](../../../../recipes/LTX/LTX-2.md) for all checkpoints,
pipeline selection, requests, defaults, and advanced options.

## Example materials

??? abstract "response.json"
    ``````json
    --8<-- "examples/online_serving/text_to_video/response.json"
    ``````
??? abstract "run_curl_ltx2.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_video/run_curl_ltx2.sh"
??? abstract "run_curl_hunyuan_video_15.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_video/run_curl_hunyuan_video_15.sh"
    ``````
??? abstract "run_curl_text_to_video.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_video/run_curl_text_to_video.sh"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_video/run_server.sh"
    ``````
??? abstract "run_server_ltx2.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_video/run_server_ltx2.sh"
??? abstract "run_server_hunyuan_video_15.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_video/run_server_hunyuan_video_15.sh"
    ``````
