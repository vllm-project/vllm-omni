# Positronic robot policy API

Use binary MessagePack over WebSocket at `/api/v1/session[/<model_id>]`.
`GET /api/v1/models` lists policy IDs; the named route also accepts the served-model alias.
Clients send observations, execute returned commands and supply the next observations.
[OpenPI](openpi_api.md) uses a separate handshake and envelope.

## Availability and launch

The adapter currently supports [Praxis-1](../models/praxis1.md#positronic-session-settings)
with one stage/worker replica. Additional policies need observation, setting and
command mappings; model guides own units, coordinate frames, rate and horizon.

```bash
vllm serve "$MODEL" --omni --host 127.0.0.1 --port 8123 \
  --served-model-name policy --deploy-config "$DEPLOY_CONFIG"
```

Set `MODEL`/`DEPLOY_CONFIG` for the policy; see the
[Praxis-1 model guide](../models/praxis1.md#usage).
With `--api-key` or `VLLM_API_KEY`, send `Authorization: Bearer <token>` on HTTP
and WebSocket handshakes. CLI keys take precedence; no key disables authentication.

## Wire contract

Each binary frame contains one MessagePack message. Connections receive:

1. `{"status": "loading", "message": "Loading policy..."}`.
2. `{"status": "ready", "meta": {...}}`.

Weights are loaded at server startup; `loading` initializes the session. Ready
metadata declares effective settings, seed, codecs and client preprocessing in
`local_stack`; use its history offsets and image restrictions.

Observations follow the model schema. Raw NumPy markers and JPEG stacks use
Positronic formats. H.264 is neither advertised nor decoded. Unknown additional
`image.*` camera keys are ignored before decoding; JPEG stacks are limited to
25 frames per marker and 25 frames per configured camera in total. Arrays and
decoded RGB share a 64 MiB budget; container sizes and total elements are
bounded during MessagePack parsing. A malformed, oversized or unsupported
observation receives `{"status": "error", "error": "Invalid observation: ..."}`
and close code 1008; the session ends. Wait for each reply before sending
another observation; pipelining closes the socket with code 1008.

Replies contain:

```text
{
  "result": [<command dictionary>, ...],
  "effective_prompt": <current task>,
  "prompt_original": null,
  "server_timing": {...}
}
```

Commands come from the policy's conversion and exclude video diagnostics.
Optional timing reports total deserialization, JPEG decoding, inference and
effective sampler settings. `image_codec` is `raw` or `jpeg`; `video_decode_ms`
is zero and `wire_jpeg_quality_est` is null (quality is not estimated).

## Connection settings

Values are JSON literals when parseable, otherwise strings. For example,
`?sample_seed=7&return_server_timing=false` fixes the seed and omits timing.

| Query | Behavior |
| --- | --- |
| `sample_seed` | Nonnegative 63-bit integer; otherwise random 31-bit base seed |
| `prompt` | Initial task, overridable by observation `task`/`prompt` |
| `max_chunks_per_episode` | Positive call budget or null |
| `return_server_timing` | Boolean, default true |
| `rollout_id` | Diagnostic label; does not identify/resume worker state |

Follow the model guide for sampler/controller/context settings and defaults;
ready metadata reports effective values. Unknown/unsupported settings receive
`{"status": "error", "error": ...}` and close code 1008 before inference.
Observations cannot override connection-owned seeds, session IDs or sampling.

## Session lifetime

Each socket owns a UUID-isolated episode while sharing loaded weights.
A worker slot is reserved before `ready`; a full worker refuses with code 1008.
Disconnect, failure or an exceeded budget releases worker state. There is no
application receive timeout; WebSocket pings handle connection liveness. Worker
history belongs to the socket until it closes; offline/OpenPI state follows
`session_idle_timeout`. Engine failures close
with code 1011; rejected client input closes with code 1008.
Reconnecting starts a new episode even with the same rollout ID. The client
owns command execution and robot control. CPU tests cover protocol/lifecycle;
model numerical, hardware and simulation evidence belongs in its guide/recipe.
