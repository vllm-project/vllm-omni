# Text-To-Audio Online Serving

This example demonstrates how to deploy text-to-audio diffusion models for
online audio generation using vLLM-Omni.

## Supported Models

| Model | Model ID | Tasks | Endpoint |
|-------|----------|-------|----------|
| Stable Audio Open | `stabilityai/stable-audio-open-1.0` | text-to-audio | `POST /v1/audio/generate` |

The online example shares the unified offline entrypoint
[`examples/offline_inference/text_to_audio/text_to_audio.py`](../../offline_inference/text_to_audio/text_to_audio.py).

## Stable Audio Open

Stable Audio Open is served through the OpenAI-compatible
`POST /v1/audio/generate` endpoint: a JSON request in, binary audio (WAV by
default) out.

> Stable Audio Open is a gated Hugging Face model. Accept the license on the
> model card and `huggingface-cli login` before downloading the checkpoint.

### Start Server

```bash
bash run_server_stable_audio.sh                 # defaults: MODEL=stabilityai/stable-audio-open-1.0, PORT=8091
```

Or directly:

```bash
vllm serve stabilityai/stable-audio-open-1.0 --omni \
    --port 8091 --gpu-memory-utilization 0.9 --trust-remote-code --enforce-eager
```

Environment overrides: `MODEL`, `PORT`.

### Send Requests (curl)

```bash
# Using the provided script (env-overridable PROMPT, AUDIO_LENGTH, SEED, OUTPUT_PATH, ...)
bash run_curl_stable_audio.sh

# Or directly
curl -sS -X POST http://localhost:8091/v1/audio/generate \
  -H "Content-Type: application/json" \
  -d '{
    "input": "A piano playing a gentle melody",
    "audio_length": 10.0,
    "negative_prompt": "Low quality, distorted, noisy",
    "guidance_scale": 7.0,
    "num_inference_steps": 100,
    "seed": 42,
    "response_format": "wav"
  }' --output stable_audio_output.wav
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | Text prompt describing the audio to generate |
| `audio_length` | float | ~47s | Audio duration in seconds (max ~47s for `stable-audio-open-1.0`) |
| `audio_start` | float | 0.0 | Audio start time in seconds |
| `negative_prompt` | string | null | Text describing what to avoid |
| `guidance_scale` | float | 7.0 | Classifier-free guidance scale |
| `num_inference_steps` | int | model default | Number of denoising steps |
| `seed` | int | null | Random seed for reproducibility |
| `response_format` | string | "wav" | Output format: `wav`, `mp3`, `flac`, `pcm`, `aac`, `opus` |

See [`docs/serving/audio_generate_api.md`](../../../docs/serving/audio_generate_api.md)
for the full API reference.
