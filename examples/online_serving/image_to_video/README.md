# Image-To-Video

This example demonstrates how to deploy the Wan2.2 image-to-video model for online video generation using vLLM-Omni.

## Start Server

### Basic Start

```bash
vllm serve Wan-AI/Wan2.2-I2V-A14B-Diffusers --omni --port 8091
```

### Start with Parameters

Or use the startup script:

```bash
bash run_server.sh
```

The script allows overriding:
- `MODEL` (default: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`)
- `PORT` (default: `8091`)
- `BOUNDARY_RATIO` (default: `0.875`)
- `FLOW_SHIFT` (default: `12.0`)
- `CACHE_BACKEND` (default: `none`)
- `ENABLE_CACHE_DIT_SUMMARY` (default: `0`)

### Ascend / Local LightX2V Example

For a local Wan2.2-LightX2V Diffusers directory on Ascend/NPU, you can start the server like this:

```bash
vllm serve /path/to/Wan2.2-I2V-A14B-LightX2V-Diffusers-Lightning \
  --omni \
  --port 8091 \
  --flow-shift 12 \
  --cfg-parallel-size 1 \
  --ulysses-degree 4 \
  --use-hsdp \
  --trust-remote-code \
  --allowed-local-media-path / \
  --seed 42
```

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
  -F "prompt=A bear playing with yarn, smooth motion" \
  -F "input_reference=@/path/to/input.png" \
  -F "size=832x480" \
  -F "num_frames=33" \
  -F "fps=16" \
  -F "negative_prompt=low quality, blurry, static" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=1.0" \
  -F "guidance_scale_2=1.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=12.0" \
  -F 'extra_params={"sample_solver":"euler"}' \
  -F "seed=42" \
  -o sync_i2v_output.mp4
```

For Wan Lightning/Distill checkpoints, pass `{"sample_solver":"euler"}` via `extra_params`. The default solver is `unipc`.

Example matching the local LightX2V deployment above:

```bash
curl -sS -X POST http://localhost:8091/v1/videos/sync \
  -H "Accept: video/mp4" \
  -F "prompt=A cat playing with yarn" \
  -F "input_reference=@/path/to/input.jpg" \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=81" \
  -F "fps=16" \
  -F "num_inference_steps=4" \
  -F "guidance_scale=1.0" \
  -F "guidance_scale_2=1.0" \
  -F "boundary_ratio=0.875" \
  -F "seed=42" \
  -F 'extra_params={"sample_solver":"euler"}' \
  -o ./output.mp4
```

Use `/v1/videos/sync` if you want to write the MP4 directly to a file. `POST /v1/videos` is async and returns job metadata, not inline `b64_json`.

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
# Basic image-to-video generation
bash run_curl_image_to_video.sh

# Wan Lightning/Distill checkpoints
SAMPLE_SOLVER=euler bash run_curl_image_to_video.sh

# Or execute directly (OpenAI-style multipart)
create_response=$(curl -s http://localhost:8091/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=A bear playing with yarn, smooth motion" \
  -F "negative_prompt=low quality, blurry, static" \
  -F "input_reference=@/path/to/qwen-bear.png" \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=33" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=1.0" \
  -F "guidance_scale_2=1.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=12.0" \
  -F 'extra_params={"sample_solver":"euler"}' \
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
curl -L "http://localhost:8091/v1/videos/${video_id}/content" -o wan22_i2v_output.mp4
```

## Request Format

### Required Fields

```bash
curl -X POST http://localhost:8091/v1/videos \
  -F "prompt=A bear playing with yarn, smooth motion" \
  -F "negative_prompt=low quality, blurry, static" \
  -F "input_reference=@/path/to/qwen-bear.png"
```

### Alternative JSON-Safe Reference Input

Use `image_reference` when you want to pass a URL or JSON-safe image reference
instead of uploading a file. Do not send `input_reference` and
`image_reference` together.

```bash
curl -X POST http://localhost:8091/v1/videos \
  -F "prompt=A bear playing with yarn, smooth motion" \
  -F 'image_reference={"image_url":"https://example.com/qwen-bear.png"}'
```

### Generation with Parameters

```bash
curl -X POST http://localhost:8091/v1/videos \
  -F "prompt=A bear playing with yarn, smooth motion" \
  -F "negative_prompt=low quality, blurry, static" \
  -F "input_reference=@/path/to/qwen-bear.png" \
  -F "width=832" \
  -F "height=480" \
  -F "num_frames=33" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=1.0" \
  -F "guidance_scale_2=1.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=12.0" \
  -F 'extra_params={"sample_solver":"euler"}' \
  -F "seed=42"
```

`sample_solver` is supported by Wan2.2 online serving through the existing `extra_params` field, which is merged into the pipeline `extra_args`. Use `unipc` for the default multistep solver, or `euler` for Lightning/Distill checkpoints.

## Create Response Format

`POST /v1/videos` returns a job record, not inline base64 video data.

```json
{
  "id": "video_gen_123",
  "object": "video",
  "status": "queued",
  "model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
  "prompt": "A bear playing with yarn, smooth motion",
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
curl -L http://localhost:8091/v1/videos/${video_id}/content -o wan22_i2v_output.mp4
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
