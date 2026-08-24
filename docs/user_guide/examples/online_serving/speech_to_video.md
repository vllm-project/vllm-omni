# Speech-To-Video

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/speech_to_video>.

This example demonstrates how to deploy the Wan2.2 speech-to-video (S2V) model for online video generation using vLLM-Omni.

## Supported Models

| Model | Model ID |
|-------|----------|
| Wan2.2 S2V (14B) | `Wan-AI/Wan2.2-S2V-14B` |

## Start Server

### Basic Start

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
vllm serve Wan-AI/Wan2.2-S2V-14B --omni \
  --model-class-name WanS2VPipeline \
  --tensor-parallel-size 2 \
  --flow-shift 3.0 \
  --vae-use-slicing --vae-use-tiling \
  --cache-backend cache_dit \
  --port 8091
```

### Start with Script

```bash
bash run_server.sh
```

The script allows overriding:
- `MODEL` (default: `Wan-AI/Wan2.2-S2V-14B`)
- `PORT` (default: `8091`)
- `FLOW_SHIFT` (default: `3.0`)
- `TP` (default: `2`)
- `CACHE_BACKEND` (default: `cache_dit`)

### Start with 4 GPUs

```bash
TP=4 bash run_server.sh
```

## API Endpoints

The server provides OpenAI-compatible video generation endpoints:

- `POST /v1/videos`: create a video generation job (async)
- `POST /v1/videos/sync`: generate a video and return raw bytes (sync, for benchmarks)
- `GET /v1/videos/{video_id}`: retrieve the current job status and metadata
- `GET /v1/videos`: list stored video jobs
- `GET /v1/videos/{video_id}/content`: download the generated video file
- `DELETE /v1/videos/{video_id}`: delete the job and any stored output

## Sync API (Recommended for Testing)

`POST /v1/videos/sync` blocks until generation completes and returns the raw
video bytes (`video/mp4`) directly in the response body.

```bash
bash run_curl_speech_to_video.sh
```

## Async Job API

`POST /v1/videos` creates an asynchronous job and returns immediately.

```bash
no_proxy=127.0.0.1 \
create_response=$(curl -sS -X POST http://127.0.0.1:8091/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=A person singing" \
  -F 'image_reference={"image_url": "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.png"}' \
  -F 'audio_reference={"audio_url": "https://raw.githubusercontent.com/Wan-Video/Wan2.2/main/examples/Five%20Hundred%20Miles.MP3"}' \
  -F "width=832" -F "height=480" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.5" \
  -F "fps=16")

video_id=$(echo "$create_response" | jq -r '.id')

# Poll until complete
while true; do
  status=$(curl -s "http://127.0.0.1:8091/v1/videos/${video_id}" | jq -r '.status')
  if [ "$status" = "completed" ]; then
    break
  fi
  if [ "$status" = "failed" ]; then
    echo "Video generation failed"
    exit 1
  fi
  sleep 2
done

# Download
curl -L "http://127.0.0.1:8091/v1/videos/${video_id}/content" -o s2v_output.mp4
```

## Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | - | Text description of the desired video |
| `image_reference` | JSON str | - | Image reference: `{"image_url": "..."}` — supports HTTP(s) URLs or base64 data URLs |
| `audio_reference` | JSON str | - | Audio reference: `{"audio_url": "..."}` — supports HTTP(s) URLs or base64 data URLs |
| `width` | int | None | Video width in pixels |
| `height` | int | None | Video height in pixels |
| `fps` | int | None | Output video frame rate |
| `num_inference_steps` | int | None | Number of denoising steps (40 recommended) |
| `guidance_scale` | float | None | CFG guidance scale |
| `seed` | int | None | Random seed for reproducibility |

## Notes

- S2V requires both a reference image (`image_reference`) and an audio reference
  (`audio_reference`). The generated video will show a person matching the reference
  image with lip movements synchronized to the audio.
- `audio_reference` accepts a JSON string: `{"audio_url": "..."}` where the URL can be:
  - An HTTP/HTTPS URL (e.g., `https://example.com/audio.mp3`)
  - A base64 data URL (e.g., `data:audio/mp3;base64,...`)
- `--model-class-name WanS2VPipeline` is required on the server to select the
  S2V pipeline (distinct from the T2V/I2V pipelines).
- `--cache-backend cache_dit` enables DiT caching for ~2x speedup on cached steps.
- Audio is muxed into the output MP4 automatically.
- Always pass `fps=16` to ensure correct video/audio alignment.
