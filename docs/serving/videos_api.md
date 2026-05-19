# Videos API

vLLM-Omni provides an OpenAI-compatible video generation API for diffusion
video models. The API supports asynchronous video jobs through `/v1/videos` and
a synchronous benchmark-oriented endpoint through `/v1/videos/sync`.

Each server instance runs a single model specified at startup with
`vllm serve <model> --omni`.

## Quick Start

### Start the Server

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --omni --port 8091
```

### Create a Video Job

```bash
create_response=$(curl -s http://localhost:8091/v1/videos \
  -F "prompt=A cinematic tracking shot of a mountain lake at sunrise" \
  -F "width=1280" \
  -F "height=720" \
  -F "num_frames=80" \
  -F "fps=16" \
  -F "num_inference_steps=40")

video_id=$(echo "${create_response}" | jq -r '.id')
```

### Poll and Download

```bash
curl -s "http://localhost:8091/v1/videos/${video_id}" | jq .
curl -L "http://localhost:8091/v1/videos/${video_id}/content" -o output.mp4
```

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/videos` | `POST` | Create an asynchronous video generation job |
| `/v1/videos/sync` | `POST` | Generate a video synchronously and return raw video bytes |
| `/v1/videos/{video_id}` | `GET` | Retrieve job status and metadata |
| `/v1/videos` | `GET` | List stored video jobs |
| `/v1/videos/{video_id}/content` | `GET` | Download generated video content |
| `/v1/videos/{video_id}` | `DELETE` | Delete a video job and stored output |

### Request Parameters

`POST /v1/videos` and `POST /v1/videos/sync` accept `multipart/form-data`.

#### OpenAI-style fields

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text prompt for video generation |
| `model` | string | server's model | Optional model name |
| `seconds` | string | null | Requested clip duration in seconds |
| `size` | string | null | Requested output size in `WIDTHxHEIGHT` format |
| `user` | string | null | Optional user identifier |

#### vLLM-Omni extension fields

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_reference` | file | null | Uploaded reference image for image-to-video requests |
| `image_reference` | string | null | JSON-encoded reference image payload; do not combine with `input_reference` |
| `width` | integer | model default | Output video width |
| `height` | integer | model default | Output video height |
| `num_frames` | integer | model default | Number of generated frames |
| `fps` | integer | model default | Output frames per second |
| `num_inference_steps` | integer | model default | Number of diffusion steps |
| `guidance_scale` | number | null | CFG guidance scale for the low-noise stage |
| `guidance_scale_2` | number | null | CFG guidance scale for the high-noise stage |
| `boundary_ratio` | number | null | Boundary split ratio for multi-stage denoising |
| `flow_shift` | number | null | Scheduler flow-shift value |
| `true_cfg_scale` | number | null | True CFG scale when supported by the model |
| `seed` | integer | null | Random seed for reproducibility |
| `negative_prompt` | string | null | Text describing what to avoid in the generated video |
| `enable_frame_interpolation` | boolean | null | Enable post-generation frame interpolation |
| `frame_interpolation_exp` | integer | null | Interpolation exponent; `1=2x`, `2=4x`, and so on |
| `frame_interpolation_scale` | number | null | RIFE inference scale |
| `frame_interpolation_model_path` | string | null | Local path or Hugging Face repo for the interpolation model |
| `lora` | string | null | JSON-encoded LoRA configuration object |
| `extra_params` | string | null | JSON-encoded object for additional model-specific parameters |

### Create Response

`POST /v1/videos` returns a job record:

```json
{
  "id": "video-123",
  "status": "queued",
  "created_at": 1701234567
}
```

The final content is available from `/v1/videos/{video_id}/content` after the
job status becomes `completed`.

### Synchronous Response

`POST /v1/videos/sync` blocks until generation finishes and returns raw video
bytes. It is useful for benchmarks and simple scripts that do not need job
storage or polling.

## Examples

### Image-to-Video

```bash
curl -s http://localhost:8091/v1/videos \
  -F "prompt=animate this image with subtle camera movement" \
  -F "input_reference=@input.png" \
  -F "width=1280" \
  -F "height=720" \
  -F "num_frames=80" \
  -F "fps=16"
```

### Synchronous Generation

```bash
curl -X POST http://localhost:8091/v1/videos/sync \
  -F "prompt=A small robot walking through a neon city" \
  -F "width=854" \
  -F "height=480" \
  -F "num_frames=80" \
  -F "fps=16" \
  -o output.mp4
```

## Storage

Set `VLLM_OMNI_STORAGE_PATH` to control where asynchronous video outputs are
stored:

```bash
export VLLM_OMNI_STORAGE_PATH=/var/tmp/vllm-omni-videos
```

## Model-Specific Examples

For complete text-to-video and image-to-video walkthroughs, see:

- [Text-to-Video](../user_guide/examples/online_serving/text_to_video.md)
- [Image-to-Video](../user_guide/examples/online_serving/image_to_video.md)
