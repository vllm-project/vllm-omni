# Audio Generate API

vLLM-Omni provides an API for text-to-audio generation using diffusion-based models such as Stable Audio.

Unlike the [Speech API](speech_api.md) which targets text-to-speech synthesis, the Audio Generate API is designed for general-purpose audio generation from text descriptions (sound effects, music, ambient soundscapes, etc.).

Each server instance runs a single model (specified at startup via `vllm-omni serve <model> --omni`).

## Quick Start

### Start the Server

```bash
vllm-omni serve stabilityai/stable-audio-open-1.0 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --omni
```

### Generate Audio

**Using curl:**

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "The sound of a cat purring",
        "audio_length": 10.0
    }' --output cat.wav
```

**Using Python:**

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/audio/generate",
    json={
        "input": "The sound of a cat purring",
        "audio_length": 10.0,
    },
    timeout=300.0,
)

with open("cat.wav", "wb") as f:
    f.write(response.content)
```

## API Reference

### Endpoint

```
POST /v1/audio/generate
Content-Type: application/json
```

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | Text prompt describing the audio to generate |
| `model` | string | server's model | Model to use (optional, should match server if specified) |
| `response_format` | string | "wav" | Audio format: wav, mp3, flac, pcm, aac, opus |
| `speed` | float | 1.0 | Playback speed (0.25 - 4.0) |

#### Diffusion Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_length` | float | null | Audio duration in seconds (default value is the max ~47s for `stable-audio-open-1.0`) |
| `audio_start` | float | 0.0 | Audio start time in seconds |
| `negative_prompt` | string | null | Text describing what to avoid in generation |
| `guidance_scale` | float | model default | Classifier-free guidance scale (higher = more adherence to prompt) |
| `num_inference_steps` | int | model default | Number of denoising steps (higher = better quality, slower) |
| `seed` | int | null | Random seed for reproducible generation |

### Response Format

Returns binary audio data with the appropriate `Content-Type` header:

| `response_format` | Content-Type |
|--------------------|--------------|
| `wav` | `audio/wav` |
| `mp3` | `audio/mpeg` |
| `flac` | `audio/flac` |
| `pcm` | `audio/pcm` |
| `aac` | `audio/aac` |
| `opus` | `audio/opus` |

## Examples

### Basic Generation

Generate audio with only a text prompt (model defaults for all other parameters):

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "The sound of ocean waves crashing on a beach"
    }' --output ocean.wav
```

### Custom Duration

Specify an explicit audio length in seconds:

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "A dog barking",
        "audio_length": 5.0
    }' --output dog_5s.wav
```

### High Quality with Negative Prompt

Use a negative prompt to steer generation away from undesired characteristics, and increase inference steps for higher quality:

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "A piano playing a gentle melody",
        "audio_length": 10.0,
        "negative_prompt": "Low quality, distorted, noisy",
        "guidance_scale": 8.0,
        "num_inference_steps": 150
    }' --output piano_hq.wav
```

### Reproducible Generation

Set a `seed` to get deterministic results across runs:

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Thunder and rain sounds",
        "audio_length": 15.0,
        "seed": 42
    }' --output thunder.wav
```

### Full Control

Combine all parameters for precise control over generation:

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Thunder and rain sounds",
        "audio_length": 15.0,
        "negative_prompt": "Low quality",
        "guidance_scale": 7.0,
        "num_inference_steps": 100,
        "seed": 42
    }' --output thunder_rain.wav
```

### Quick Generation (Fewer Steps)

For faster generation with slightly lower quality:

```bash
curl -X POST http://localhost:8000/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Birds chirping in a forest",
        "audio_length": 8.0,
        "num_inference_steps": 50
    }' --output birds_quick.wav
```

### Python Client

```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/audio/generate",
    json={
        "input": "Thunder and rain",
        "audio_length": 15.0,
        "negative_prompt": "Low quality",
        "guidance_scale": 7.0,
        "num_inference_steps": 100,
        "seed": 42,
        "response_format": "wav",
    },
    timeout=300.0,
)

with open("thunder.wav", "wb") as f:
    f.write(response.content)
```

## Parameter Tuning Guide

### `guidance_scale`

Controls how closely the generated audio follows the text prompt.

| Range | Behaviour |
|-------|-----------|
| 3 - 5 | More creative / varied output |
| 7 (default) | Balanced adherence |
| 10+ | Strict adherence to the prompt |

### `num_inference_steps`

Controls the number of denoising steps in the diffusion process.

| Steps | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| 50 | Good | Fast | Quick previews |
| 100 | Very Good | Medium | General purpose |
| 150+ | Excellent | Slow | Final / critical audio |

### `audio_length`

Duration of the generated audio clip. For `stable-audio-open-1.0`, the maximum is approximately 47 seconds. If omitted, the model uses its own default length.

### `negative_prompt`

Describes characteristics to avoid. Common negative prompts include:

- `"Low quality, distorted, noisy"`
- `"Silence, static"`
- `"Music"` (when generating sound effects only)

## Supported Models

| Model | Description |
|-------|-------------|
| `stabilityai/stable-audio-open-1.0` | Open-source audio generation model, up to ~47 seconds, 44.1 kHz stereo |

## Error Responses

### 400 Bad Request

Invalid or missing parameters:

```json
{
    "error": {
        "message": "Audio generation model did not produce audio output.",
        "type": "BadRequestError",
        "param": null,
        "code": 400
    }
}
```

### 404 Not Found

Model mismatch:

```json
{
    "error": {
        "message": "The model `xxx` does not exist.",
        "type": "NotFoundError",
        "param": "model",
        "code": 404
    }
}
```

### 422 Unprocessable Entity

Pydantic validation failure (e.g. invalid `response_format`, `speed` out of range):

```json
{
    "detail": [
        {
            "type": "literal_error",
            "msg": "Input should be 'wav', 'pcm', 'flac', 'mp3', 'aac' or 'opus'",
            ...
        }
    ]
}
```

## Troubleshooting

### "Audio generation model did not produce audio output"

The model finished but returned no audio data. Verify the server started successfully and the model loaded without errors.

### Server Not Responding

```bash
# Check if the server is healthy
curl http://localhost:8000/health
```

### Audio Quality Issues

- Increase `num_inference_steps` (e.g. 150).
- Add a negative prompt: `"Low quality, distorted, noisy"`.
- Increase `guidance_scale` for stronger prompt adherence.

### Generation Timeout

- Reduce `num_inference_steps`.
- Reduce `audio_length`.
- Check GPU memory with `nvidia-smi`.

### Out of Memory

- Lower `--gpu-memory-utilization` (e.g. 0.8).
- Reduce `audio_length`.

## Development

Enable debug logging:

```bash
vllm-omni serve stabilityai/stable-audio-open-1.0 \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --omni \
    --uvicorn-log-level debug
```
