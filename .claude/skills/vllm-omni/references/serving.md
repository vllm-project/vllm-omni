# Serving Omni Models

API docs: `docs/serving/*.md`. Getting started: `docs/getting_started/quickstart.md`.

## Launch

```bash
vllm serve <model> --omni --port 8091 [--deploy-config path.yaml]
```

`vllm-omni serve <model> --omni ...` is equivalent. Without `--omni` the plain
vLLM server starts.

| Flag | Default | Purpose |
|------|---------|---------|
| `--deploy-config` | auto | Deploy YAML |
| `--stage-overrides` | – | JSON per-stage overrides |
| `--async-chunk` / `--no-async-chunk` | YAML | Streaming hand-off |
| `--stage-init-timeout` / `--init-timeout` | 300 / 600 s | Startup timeouts |
| `--log-stats` | off | Per-request stage and transfer timing |
| `--enable-orch-monitor` | off | Orchestrator time series |
| `--allowed-local-media-path` | – | Allow local file media in requests |
| `--worker-backend` | `multi_process` | Or `ray` |
| `--omni-lb-policy` | `random` | Replica load balancing (`round-robin`, `least-queue-length`) |
| `--stage-id`, `--headless`, `--omni-master-address/port` | – | Disaggregated / multi-node stages |

Health: `GET /health`, `GET /v1/models`.

## Online APIs relevant to omni models

| Endpoint | Use | Key fields |
|----------|-----|-----------|
| `POST /v1/chat/completions` | Multimodal in, text and/or audio out | `messages` (image/audio/video parts), `modalities`: `["text"]`, `["audio"]`, `["text","audio"]`, `stream` |
| `POST /v1/audio/speech` | TTS-style | `input`, `voice`, `response_format`, `stream`, `ref_audio`/`ref_text` |
| `WS /v1/realtime` | Realtime turn or duplex (on duplex servers) | See `docs/serving/realtime_duplex_api.md` |
| `WS /v1/realtime?duplex=1`, `WS /v1/duplex` | Full-duplex speech-to-speech | `docs/serving/full_duplex_api.md` |
| `WS /v1/video/chat/stream` | Live video understanding | `docs/serving/video_stream_api.md` |

Minimal text + audio request:

```bash
curl -s http://localhost:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "<model>",
       "messages": [{"role": "user", "content": "Introduce yourself briefly."}],
       "modalities": ["text", "audio"]}'
```

Non-streaming responses put base64 WAV in `message.audio.data`; streaming
responses emit base64 audio chunks as `delta.content` on audio choices
(`vllm_omni/entrypoints/openai/serving_chat.py`). Use `"modalities": ["text"]` to skip talker
and code2wav when isolating thinker behavior.

## Offline

```python
from vllm import SamplingParams
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="Qwen/Qwen3-Omni-30B-A3B-Instruct")  # optional deploy_config=...
outputs = omni.generate(prompts, sampling_params_list=[SamplingParams(...), ...])
omni.close()
```

One `SamplingParams` per stage, in stage order. Reference script:
`examples/offline_inference/qwen3_omni/end2end.py`. Async: `AsyncOmni(...).generate(...)`.

## Mode caveats

- A duplex deploy (`session_mode: duplex`) serves sessions over WebSocket; use a
  turn deploy for ordinary HTTP chat (MiniCPM-o: `minicpmo_4_5_turn.yaml`).
- Some realtime routes are unavailable with `async_chunk` on; check the model
  recipe under `recipes/` before promising an endpoint.
