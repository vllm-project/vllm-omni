# AudioX online serving

Launches the `AudioXPipeline` behind vLLM-Omni's OpenAI-compatible chat endpoint and provides a
minimal Python client that covers all six tasks (`t2a`, `t2m`, `v2a`, `v2m`, `tv2a`, `tv2m`).

## Start the server

```bash
cd examples/online_serving/audiox
bash run_server.sh                 # defaults: MODEL=zhangj1an/AudioX, PORT=8099
```

Environment overrides: `MODEL`, `PORT`, `DIFFUSION_ATTENTION_BACKEND`.

## Call from Python

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

- `num_inference_steps`, `guidance_scale`, `seed` as first-class OpenAI chat-completion fields
- `audiox_task`, `seconds_start`, `seconds_total`, `sigma_min`, `sigma_max` nested under
  `extra_args` (a reserved dict on the request body that the server forwards verbatim into
  the pipeline's `sampling_params.extra_args` — the same escape hatch `serving_video.py` exposes
  as `extra_params` on /v1/videos)
- For `v2*` / `tv2*` tasks, the video as a `video_url` content item (data URI for local files)

## curl

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
