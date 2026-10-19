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
(`--api-server-count 1`, the default). The Live overlays set
`session_mode: turn`, so the `/v1/realtime` full-duplex stack stays off.

```yaml
live_session_config:
  session_lifetime_s: 1800   # hard limit; fills expires_at
  max_sessions: null         # concurrent Live sessions on this API server; null is unbounded
```

On PersonaPlex, `max_sessions` also sizes the Stage 0 streaming Mimi encoder
and the Code2Wav decoders (one Mimi instance per session), so it must be set.

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
session starts get `session_not_started`. Non-null `store`, `client` and
`delegation` are `unsupported_session_config`.

When `max_sessions` sessions are active, a `session.start` gets an `error`
(`type: "server_error"`, `code: "session_limit_reached"`, `client_event_id`
set) and the socket closes with 1013 (Try Again Later); no `session.closed` is
sent.

### Server events

| Event | Notes |
| ------- | ------- |
| `session.output_audio.delta` | As the model produces it. |
| `session.output_transcript.delta` | The model's own text, timestamped with the input unit that produced it (MiniCPM-o only). |
| `session.usage.updated` | Every 60 s of session time and at close. |
| `session.closed` | `close_requested` or `expired` (`expires_at` reached). |

No input transcripts are produced.

## Context

When the next unit no longer fits the stage-0 context, the server trims the
history and resubmits it as a new request, the unit that did not fit included:

- MiniCPM-o keeps the system prompt and reference voice plus the newest units
  that fill half of the remaining context; the dropped units' text is carried
  in the system prompt (`previous: ...`).
- PersonaPlex restarts from the persona and the newest frame; its conversation
  memory resets.

## Model notes

- **MiniCPM-o 4.5**: voice `default`. Follows MiniCPM-o-Demo's greedy duplex
  loop: the checkpoint's streaming mel processor and stateful streaming audio
  encoder per session, and the Demo's decode rules
  (`MiniCPMOLiveLogitsProcessor`, installed on stage 0 by the Live overlay).
  `instructions` and text `input` become the duplex system prompt, with the
  checkpoint's `assets/HT_ref_audio.wav` as reference voice. Each session holds
  an audio-encoder cache of up to ~150 MB on the Stage 0 GPU that vLLM's
  memory profiling does not count.
- **PersonaPlex**: voices are the bundled voice prompts (`NATF2`, ...);
  `instructions` is the persona. Output audio trails the input by one extra
  frame (80 ms): a frame's agent codes are produced when the next frame
  arrives. No output transcripts.

## Example

`examples/online_serving/live_sessions/livekit/` runs a LiveKit Agents voice
agent (`GPTLiveModel`) against this endpoint.

## Metrics

`vllm_omni:live_active_sessions`, `vllm_omni:live_rejected_sessions`,
`vllm_omni:live_audio_seconds{direction}`, `vllm_omni:live_context_trims` and
`vllm_omni:live_errors{reason}`, labelled by `model_name`.
