# Standalone Experimental Servers

Most vLLM-Omni endpoints run on the unified server started with
`vllm serve <model> --omni`. The experimental package also contains one
separate user-facing server process; its routes are not added to the unified
API server.

| Server | Transport | Role |
| --- | --- | --- |
| JoyVL interaction server | HTTP | Adds state, memory, proactive decisions, and delegation in front of a separate OpenAI-compatible model backend |

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

python -m vllm_omni.experimental.fullduplex.joyvl.serving.server \
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
