# Text-To-Audio Online Serving

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/text_to_audio>.


This example demonstrates how to deploy text/video-to-audio diffusion models for
online audio generation using vLLM-Omni.

## Supported Models

| Model | Model ID | Tasks | Endpoint |
|-------|----------|-------|----------|
| Stable Audio Open | `stabilityai/stable-audio-open-1.0` | text-to-audio | `POST /v1/audio/generate` |
| AudioX | `zhangj1an/AudioX` | `t2a` / `t2m` / `v2a` / `v2m` / `tv2a` / `tv2m` | `POST /v1/chat/completions` |

The two models use different serving APIs, so each has its own server and curl
script. Both share the unified offline entrypoint
[`examples/offline_inference/text_to_audio/text_to_audio.py`](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_audio/text_to_audio.py).

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
| `response_format` | string | "wav" | Output format: `wav`, `mp3`, `flac`, `pcm`, `opus` |

See [`docs/serving/audio_generate_api.md`](https://github.com/vllm-project/vllm-omni/tree/main/docs/serving/audio_generate_api.md)
for the full API reference.

## AudioX

AudioX is served through the standard OpenAI chat-completions endpoint and
requires an explicit pipeline class at launch. Per-request task and sampler
knobs (declared in
[`vllm_omni/model_extras/audiox.py`](https://github.com/vllm-project/vllm-omni/tree/main/vllm_omni/model_extras/audiox.py))
are sent under `extra_body`. The response carries base64 WAV in
`choices[0].message.audio.data`.

### Start Server

```bash
bash run_server_audiox.sh                 # defaults: MODEL=zhangj1an/AudioX, PORT=8099
```

Or directly:

```bash
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
  vllm serve zhangj1an/AudioX --omni --model-class-name AudioXPipeline --port 8099
```

Environment overrides: `MODEL`, `PORT`, `DIFFUSION_ATTENTION_BACKEND`.

### Send Requests (curl)

```bash
# text-to-audio (default TASK=t2a)
bash run_curl_audiox.sh

# text-to-music
TASK=t2m PROMPT="Uplifting ukulele tune for a travel vlog" bash run_curl_audiox.sh

# text+video-to-audio (v2*/tv2* require VIDEO; local files are inlined as a data URI)
TASK=tv2a PROMPT="drum beating sound and human talking" \
  VIDEO=https://zeyuet.github.io/AudioX/static/samples/V2M/1XeBotOFqHA.mp4 \
  bash run_curl_audiox.sh
```

Or directly:

```bash
curl -sS -X POST http://localhost:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "zhangj1an/AudioX",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Uplifting ukulele"}]}],
    "extra_body": {
      "num_inference_steps": 250,
      "guidance_scale": 7.0,
      "seed": 42,
      "audiox_task": "t2m",
      "seconds_total": 10.0,
      "sigma_min": 0.03,
      "sigma_max": 1000.0
    }
  }' > t2m.json
```

`extra_body` knobs (declared in `vllm_omni/model_extras/audiox.py`):

| Key | Description |
|-----|-------------|
| `audiox_task` | One of `t2a` / `t2m` / `v2a` / `v2m` / `tv2a` / `tv2m` |
| `num_inference_steps` | Number of denoising steps |
| `guidance_scale` | Classifier-free guidance scale |
| `seed` | Random seed for reproducibility |
| `seconds_start` | Audio start offset in seconds |
| `seconds_total` | Audio duration in seconds (fixed ~10s for the upstream bundle) |
| `sigma_min` / `sigma_max` | Sampler sigma range |

For `v2*` / `tv2*` tasks, attach the video as a `video_url` content item (a
`data:video/mp4;base64,...` URI for local files, or an http(s) URL).

## Example materials

??? abstract "run_curl_audiox.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_audio/run_curl_audiox.sh"
    ``````
??? abstract "run_curl_stable_audio.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_audio/run_curl_stable_audio.sh"
    ``````
??? abstract "run_server_audiox.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_audio/run_server_audiox.sh"
    ``````
??? abstract "run_server_stable_audio.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_audio/run_server_stable_audio.sh"
    ``````
