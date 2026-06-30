# Text-To-Audio (Online Serving)

vLLM-Omni serves diffusion-based audio generation models through two
endpoints, depending on the pipeline:

- **Stable Audio** — OpenAI-compatible [`POST /v1/audio/generate`](../../../docs/serving/audio_generate_api.md).
- **AudioX** — OpenAI-compatible `POST /v1/chat/completions` with task and
  generation knobs passed under `extra_args`.

Each model has its own subdirectory containing client snippets and helper
scripts; this README is the single doc entry point for online serving of
all of them.

For offline inference, see [`examples/offline_inference/text_to_audio/`](../../offline_inference/text_to_audio/README.md).
For the full list of supported architectures across all modalities, see
[Supported Models](../../../docs/models/supported_models.md).

For text-to-speech (autoregressive TTS) online serving, see [`examples/online_serving/text_to_speech/`](../text_to_speech/README.md).

## Supported Models

| Model | HuggingFace repo | Endpoint | Tasks | Sample rate |
|---|---|---|---|---|
| Stable Audio Open | `stabilityai/stable-audio-open-1.0` | `/v1/audio/generate` | `t2a` | 44.1 kHz |
| AudioX | `zhangj1an/AudioX` | `/v1/chat/completions` | `t2a`, `t2m`, `v2a`, `v2m`, `tv2a`, `tv2m` | 48 kHz (default) |

---

## Stable Audio Open

OpenAI-compatible text-to-audio via `/v1/audio/generate`.

### Launch

```bash
vllm-omni serve stabilityai/stable-audio-open-1.0 \
    --host 0.0.0.0 \
    --port 8091 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enforce-eager \
    --omni
```

### Generate via curl

```bash
curl -X POST http://localhost:8091/v1/audio/generate \
    -H "Content-Type: application/json" \
    -d '{
        "input": "The sound of a cat purring",
        "audio_length": 10.0
    }' --output cat.wav
```

### Python client

```bash
python examples/online_serving/text_to_audio/stable_audio/stable_audio_client.py \
    --text "The sound of a cat purring" \
    --audio_length 10.0 \
    --output cat.wav
```

### Bash script (multiple example payloads)

```bash
bash examples/online_serving/text_to_audio/stable_audio/curl_examples.sh
```

### API reference

Endpoint:

```
POST /v1/audio/generate
```

Request body:

```json
{
    "input": "Text description of the audio",
    "audio_length": 10.0,
    "audio_start": 0.0,
    "negative_prompt": "Low quality",
    "guidance_scale": 7.0,
    "num_inference_steps": 100,
    "seed": 42,
    "response_format": "wav"
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | **required** | Text prompt describing the audio to generate |
| `audio_length` | float | ~47 s | Audio duration in seconds (max ~47 s for `stable-audio-open-1.0`) |
| `audio_start` | float | 0.0 | Audio start time in seconds |
| `negative_prompt` | string | null | Text describing what to avoid in generation |
| `guidance_scale` | float | 7.0 | Classifier-free guidance scale (higher = more adherence to prompt) |
| `num_inference_steps` | int | 50 | Number of denoising steps (higher = better quality, slower) |
| `seed` | int | null | Random seed for reproducibility |
| `response_format` | string | `"wav"` | Output format: `wav`, `mp3`, `flac`, `pcm` |

Response: audio data in the requested format (default: WAV).

### Tips

- Keep `audio_length` under 47 seconds for `stable-audio-open-1.0`.
- Quality vs. speed: 50 steps (fast) / 100 steps (balanced) / 150+ (high quality).
- Guidance scale: lower (3-5) for variety, 7 for balance, 10+ for prompt literal.
- Use the same `seed` for reproducible results.

### Troubleshooting

- **Server not responding** — `curl http://localhost:8091/health` and check server logs.
- **Audio quality issues** — raise `num_inference_steps`, add `negative_prompt`, raise `guidance_scale`.
- **Generation timeout** — lower `num_inference_steps` or `audio_length`; check `nvidia-smi`.
- **Wrong audio length** — verify `audio_length` is within model limits and adjust `audio_start` if trimming.

### See also

- [Stable Audio model card](https://huggingface.co/stabilityai/stable-audio-open-1.0)
- [Audio generation API reference](../../../docs/serving/audio_generate_api.md)
- [Offline Stable Audio example](../../offline_inference/text_to_audio/stable_audio/README.md)

---

## AudioX

Launches `AudioXPipeline` behind vLLM-Omni's OpenAI-compatible chat
endpoint. The Python client covers all six tasks (`t2a`, `t2m`, `v2a`,
`v2m`, `tv2a`, `tv2m`).

### Launch

```bash
cd examples/online_serving/text_to_audio/audiox
bash run_server.sh                 # defaults: MODEL=zhangj1an/AudioX, PORT=8099
```

Environment overrides: `MODEL`, `PORT`, `DIFFUSION_ATTENTION_BACKEND`.

### Sending requests

```bash
# text-to-audio
python openai_chat_client.py --task t2a \
    --prompt "Fireworks burst twice, followed by a period of silence before a clock begins ticking." \
    --output t2a.wav

# text-to-music
python openai_chat_client.py --task t2m \
    --prompt "Uplifting ukulele tune for a travel vlog" \
    --output t2m.wav

# video-to-audio (no text)
python openai_chat_client.py --task v2a --video path/to/clip.mp4 --output v2a.wav

# text+video-to-audio
python openai_chat_client.py --task tv2a \
    --prompt "drum beating sound and human talking" \
    --video path/to/clip.mp4 \
    --output tv2a.wav
```

The client sends:

- `num_inference_steps`, `guidance_scale`, `seed` as first-class OpenAI
  chat-completion fields.
- `audiox_task`, `seconds_start`, `seconds_total`, `sigma_min`, `sigma_max`
  nested under `extra_args` (a reserved dict on the request body that the
  server forwards verbatim into the pipeline's `sampling_params.extra_args`
  — the same escape hatch `serving_video.py` exposes as `extra_params`
  on `/v1/videos`).
- For `v2*` / `tv2*` tasks, the video as a `video_url` content item
  (data URI for local files).

### curl

```bash
curl -sS -X POST http://localhost:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "zhangj1an/AudioX",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Uplifting ukulele"}]}],
    "num_inference_steps": 250,
    "guidance_scale": 7.0,
    "seed": 42,
    "extra_args": {
      "audiox_task": "t2m",
      "seconds_total": 10.0,
      "sigma_min": 0.3,
      "sigma_max": 500.0
    }
  }' > t2m.json
```
