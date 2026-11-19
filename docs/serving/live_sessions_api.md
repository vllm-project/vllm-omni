# OpenAI Live Sessions API

`WS /v1/live/sessions` serves the [OpenAI Live API](https://developers.openai.com/api/docs/guides/live)
(primary WebSocket transport) for native full-duplex speech-to-speech models.
The model hears the input continuously and decides, unit by unit, when to
listen and when to speak; there is no server-side VAD, turn detection or
barge-in handling. Turn-based models are served by `/v1/realtime`.

| Model | Unit | Deploy overlay |
| ------- | ------ | ---------------- |
| MiniCPM-o 4.5 | official streaming chunk (1035 ms, then 1000 ms) | `minicpmo_4_5_live.yaml` |
| PersonaPlex | 80 ms frame | `personaplex_live.yaml` |

Each session is one vLLM resumable request: every input unit is a streaming
update, and the model's output for it is one segment.

## Deployment

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni --trust-remote-code --port 8000 \
  --deploy-config vllm_omni/deploy/minicpmo_4_5_live.yaml
```

The endpoint is enabled only when the model declares a `live_session_config`
in its pipeline and the deploy config carries a `live_session_config` object.
Otherwise the WebSocket is accepted, answered with an `error`, and closed with
code 1008. Live sessions need a single API server process
(`--api-server-count 1`, the default): `max_sessions` is counted per process,
so with a Live deploy config the server refuses to start with any other count.
Live with several API server processes is unsupported and its behaviour
undefined. The Live overlays set
`session_mode: turn`, so the `/v1/realtime` full-duplex stack stays off.

```yaml
live_session_config:
  session_lifetime_s: 1800   # hard limit; fills expires_at
  max_sessions: 8            # concurrent Live sessions on the server
  max_input_backlog_s: 10    # input audio queued behind the model before the session closes
```

All three keys are required. `max_sessions` also sizes per-session model
state: on PersonaPlex, the Stage 0 streaming Mimi encoder and the Code2Wav
decoders (one Mimi instance per session).

Size `max_sessions` so that every admitted session fits in Stage 0's KV cache
at its context length. Sessions that don't fit stall when Stage 0 runs out of
KV cache: a waiting Live request keeps its blocks and vLLM does not preempt it,
so the stalled sessions are ended by `max_input_backlog_s` (see below). The
MiniCPM-o overlay admits 8 sessions with a 16,384-token Stage 0 context, about
18 GiB of KV cache, for one 96 GB GPU; lower `max_sessions` or the Stage 0
`max_model_len` on smaller GPUs.

A model that doesn't trim its context (PersonaPlex) must hold a whole session
in Stage 0: the server refuses to start when Stage 0's `max_model_len` is less
than `session_lifetime_s` of real-time input plus the model's largest prefill.

## Protocol

Messages are JSON text frames validated against the `openai.types.live`
models. Audio is base64 raw audio in the session format with no container:
`audio/pcm` at 16000 or 24000 Hz (default 24000), `audio/pcmu`, or
`audio/pcma`. The same format applies to both directions.

### Client events

| Event | Behaviour |
| ------- | ----------- |
| `session.start` | Must be first. `model` must be the served model name. `audio.output.voice` is a model voice id (OpenAI voice names are rejected). Replies `session.started`, then `info` (`code: "capabilities"`). Invalid configuration: `error`, then close 1008. |
| `session.update` | Nothing can change after `session.start`; any non-null field is `unsupported_session_config`. An empty update replies `session.updated`. |
| `session.input_audio.append` | Advances the session clock by the chunk's duration. Malformed audio: `error` `invalid_audio`. |
| `session.input_audio.mute` / `.unmute` | While muted the model is fed silence. Replies `.muted` / `.unmuted`. |
| `session.close` | Replies `session.usage.updated` and `session.closed` (`reason: "close_requested"`), then closes 1000. |
| `session.commentary.append` | Accepted and ignored (no reply). |

`session.instructions.append`, `session.thinking.append`,
`response.item.create`, `response.create` and unknown events get `error` with
`code: "unsupported_event"`. Events other than `session.start` before the
session starts get `session_not_started`. Non-null `store`, `client`,
`delegation.client` and `delegation.responses.tools` are
`unsupported_session_config`. A `delegation` of type `responses` that
delegates nothing (as LiveKit's `GPTLiveModel` sends) is accepted and echoed;
its `model` must be the served model.

When `max_sessions` sessions are active, a `session.start` gets an `error`
(`type: "server_error"`, `code: "session_limit_reached"`, `client_event_id`
set) and the socket closes with 1013 (Try Again Later); no `session.closed` is
sent.

A running session the server can't keep going is ended with an `error`
(`type: "server_error"`, `code: "internal_server_error"`) and close 1013; the
client may start a new session. This happens when more than
`max_input_backlog_s` of input audio is queued behind the model (the client
sends faster than real time, or the server falls behind), when its stage 0
request was preempted and lost its context, or when a model that doesn't trim
fills its context. The cause is not sent to the client; the server logs it and
counts it in `vllm_omni:live_errors{reason}` (`backlog`, `preempted`,
`context`).

A session that fails on something unexpected, or on a cause a new session
would hit again, gets the same `error` and close 1011. That includes a context
that can't hold the session prefix and one unit, and a trim that can't drop
anything; both count as `live_errors{reason="context"}`.

### Server events

| Event | Notes |
| ------- | ------- |
| `session.output_audio.delta` | As the model produces it. |
| `session.output_transcript.delta` | The model's own text, timestamped with the input unit that produced it (MiniCPM-o only). |
| `session.usage.updated` | Every 60 s of session time and at close. |
| `session.closed` | `close_requested` or `expired` (`expires_at` reached). |

No input transcripts are produced.

## Context

Each model decides how it handles a full stage-0 context.

- **MiniCPM-o** trims. When the next unit no longer fits, the server keeps the
  system prompt and reference voice plus the newest units that fill half of
  the remaining context, carries the dropped units' text in the system prompt
  (`previous: ...`), and resubmits the history as a new request, the unit that
  did not fit included.
- **PersonaPlex** doesn't trim. As in the reference model, Stage 0 attends to
  a sliding window of the last 3,000 frames (240 s), so a session runs for its
  whole lifetime with about one window of KV cache; the persona and voice
  prompt slide out of the window like the rest. The Live overlay sizes Stage 0
  `max_model_len` for a whole session. A session that still overruns it (a
  client sending faster than real time) ends with close 1013.

## Model notes

- **MiniCPM-o 4.5**: voice `default`. Follows MiniCPM-o-Demo's greedy duplex
  loop: the checkpoint's streaming mel processor and stateful streaming audio
  encoder per session, and the Demo's decode rules
  (`MiniCPMOLiveLogitsProcessor`, installed on stage 0 by the Live overlay).
  `instructions` and text `input` become the duplex system prompt, with the
  checkpoint's `assets/HT_ref_audio.wav` as reference voice. Each session holds
  an audio-encoder cache of up to ~150 MB on the Stage 0 GPU that vLLM's
  memory profiling does not count. The Live overlay sets Stage 0
  `max_model_len: 16384`.
- **PersonaPlex**: voices are the bundled voice prompts (`NATF2`, ...);
  `instructions` is the persona, at most 1,024 tokens including the
  `input` messages (`invalid_value` otherwise). The Live overlay sets Stage 0
  `max_model_len` and the Helium `max_position_embeddings` to 24,000 frames:
  1,800 s of input plus the largest prefill. Output audio trails the input by one extra
  frame (80 ms): a frame's agent codes are produced when the next frame
  arrives. No output transcripts.

## Example

`examples/online_serving/live_sessions/livekit/` runs a LiveKit Agents voice
agent (`GPTLiveModel`) against this endpoint.

## Metrics

`vllm_omni:live_active_sessions`, `vllm_omni:live_rejected_sessions`,
`vllm_omni:live_audio_seconds{direction}`, `vllm_omni:live_context_trims` and
`vllm_omni:live_errors{reason}`, labelled by `model_name`. `live_errors`
reasons include `backlog`, `preempted` and `context`.
