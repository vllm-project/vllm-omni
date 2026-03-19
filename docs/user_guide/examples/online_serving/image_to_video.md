# Image-To-Video

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/image_to_video>.


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
# Basic image-to-video generation
bash run_curl_image_to_video.sh

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
  -F "seed=42"
```

## Create Response Format

`POST /v1/videos` returns a job record.

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

## Example materials

??? abstract "run_curl_image_to_video.sh"
    ``````sh
    --8<-- "examples/online_serving/image_to_video/run_curl_image_to_video.sh"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/image_to_video/run_server.sh"
    ``````
