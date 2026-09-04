# Standalone Experimental Servers

Most vLLM-Omni endpoints run on the unified server started with
`vllm serve <model> --omni`. The experimental full-duplex package also contains
two separate user-facing server processes. Their routes are not added to the
unified API server.

| Server | Transport | Role |
|--------|-----------|------|
| PersonaPlex compatibility server | WebSocket | Hosts the official PersonaPlex browser protocol and a raw-PCM alternative |
| JoyVL interaction server | HTTP | Adds state, memory, proactive decisions, and delegation in front of a separate OpenAI-compatible model backend |

## PersonaPlex Compatibility Server

Use this server when a client requires the official Moshi/PersonaPlex wire
protocol or the bundled browser UI. For the engine-owned unified duplex path,
use [`WS /v1/realtime?duplex=1`](full_duplex_api.md) instead.

Start it with:

```bash
HF_TOKEN=... CUDA_VISIBLE_DEVICES=0 python -m \
  vllm_omni.experimental.fullduplex.personaplex.serving.server \
  --port 8091
```

| Route | Purpose |
|-------|---------|
| `GET /` | Official PersonaPlex browser client, when its assets are available |
| `GET /health` | Readiness check |
| `WS /api/chat` | Moshi-compatible tagged binary protocol with Opus audio |
| `WS /v1/audio/duplex` | Simpler float32 raw-PCM protocol |

### `/api/chat` Protocol

The first byte of every binary frame is a message tag:

| Direction | Tag | Payload |
|-----------|-----|---------|
| Server to client | `0x00` | Ready handshake |
| Client to server | `0x01` | Opus microphone audio at 24 kHz |
| Server to client | `0x01` | Opus response audio |
| Server to client | `0x02` | UTF-8 inner-monologue text |

The optional `text_prompt` and `voice_prompt` query parameters select a persona
and voice. The default server admits one conversation; use `--batch-size N` to
configure multiple lockstep slots. Connections beyond capacity close with
WebSocket code 1013.

### `/v1/audio/duplex` Protocol

Send `{"type":"open","persona":"...","voice":"..."}` as a JSON text frame,
then exchange binary float32 PCM frames. The server returns generated float32
audio as binary frames and text as `{"type":"text","text":"..."}`. Send
`{"type":"close"}` and wait for `{"type":"done"}` to flush and finish.

See the [PersonaPlex example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/personaplex)
for the browser and headless clients.

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
|------------------|---------|
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
