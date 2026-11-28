# Standalone Servers

Most vLLM-Omni endpoints run on the unified server started with
`vllm serve <model> --omni`. Two standalone entrypoints provide separate
user-facing server processes; their routes are not added to the unified API
server.

| Server | Transport | Role |
| --- | --- | --- |
| JoyVL interaction server | HTTP | Adds state, memory, proactive decisions, and delegation in front of a separate OpenAI-compatible model backend |
| Mage-VL server | WebSocket | Runs codec-window input and proactive gate evaluation against a Transformers backend |

PersonaPlex no longer ships a standalone compatibility server: it is served
through the unified duplex path (`vllm_omni/deploy/personaplex.yaml`) over
[`WS /v1/realtime?duplex=1`](full_duplex_api.md); see the
[PersonaPlex example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/personaplex)
for its clients.

## JoyVL Interaction Server

JoyVL is an orchestration layer, not a model-serving engine. It calls a
separate OpenAI-compatible backend and adds per-session frame history, memory,
persona policy, and optional delegation.

Start the model backend and orchestrator separately:

```bash
vllm serve jdopensource/JoyAI-VL-Interaction-Preview \
  --served-model-name JoyAI-VL-Interaction-Preview \
  --port 8092 \
  --max-model-len 131072 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":256,"video":1}'

python -m vllm_omni.entrypoints.joyvl.serving.server \
  --port 8091 \
  --main-backend-url http://127.0.0.1:8092/v1 \
  --main-model JoyAI-VL-Interaction-Preview
```

| Method and route | Purpose |
| ------------------ | --------- |
| `GET /health` | Readiness check |
| `GET /v1/models` | Reports the configured interaction model |
| `POST /v1/chat/completions` | Processes one frame or interaction tick |
| `POST /reset` | Resets a session |
| `POST /v1/streaming/reset` | Alias for `/reset` |
| `POST /v1/streaming/persona` | Changes the session persona |

Send one multimodal Chat Completions request per video frame, normally around
one frame per second, and identify the session with `x-session-id`. Responses
include an `interaction` block whose action is `silence`, `response`, or
`delegate`.

```bash
curl http://127.0.0.1:8091/v1/chat/completions \
  -H 'content-type: application/json' \
  -H 'x-session-id: demo' \
  -d '{
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "Alert me if a fire appears"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ]}]
  }'
```

See the [JoyVL recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/JD/JoyAI-VL-Interaction.md)
for memory, personas, delegation backends, and the browser UI.

## Mage-VL Server

Start `python -m vllm_omni.entrypoints.mage_vl.serving.server` to serve
`WS /v1/mage-vl/duplex`. Its session runtime and adapter live under
`vllm_omni/entrypoints/mage_vl/`; native vLLM model execution remains follow-up
work. See the [Mage-VL recipe](../../recipes/Microsoft/Mage-VL.md) for backend,
authentication, and client options.

## Python Module Locations

The former experimental full-duplex package has been removed. Use these owners:

| Component | Module |
| --- | --- |
| JoyVL HTTP interaction service | `vllm_omni.entrypoints.joyvl.serving.server` |
| Mage-VL WebSocket service | `vllm_omni.entrypoints.mage_vl.serving.server` |
| JoyAI action parsing and prompts shared with the native pipeline | `vllm_omni.model_executor.models.joyai_vl_interaction.duplex` |
| Realtime benchmark probe | `vllm_omni.clients.realtime` |
| Camera-frame tiling | `vllm_omni.clients.video_stacking` |

The old JoyVL scaffold adapter has been retired in favor of
`JoyAIDuplexPlugin` in the native JoyAI pipeline. The HTTP orchestrator retains
its external-backend memory and delegation behavior. Applications should use
`vllm_omni.clients.duplex.DuplexClient`; the probe client retains its explicit
session handshake and media clock for benchmark and smoke-test workloads.
