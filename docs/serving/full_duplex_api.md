# Full-Duplex WebSocket API

vLLM-Omni provides a full-duplex runtime for models that can
continue receiving speech while producing speech. It adds persistent session
state, model-specific turn policy, overlap handling, playback acknowledgement,
and optional session resume.

Full duplex is distinct from the turn-based [Realtime Audio API](realtime_api.md).

This page is the endpoint overview. For the complete wire contract, the
`vllm_omni.clients.duplex.DuplexClient` Python library, and the per-model
capability gates, see the [Realtime Duplex API guide](realtime_duplex_api.md).

## Choose an Endpoint

| Endpoint | Protocol | Recommended use |
| --- | --- | --- |
| `WS /v1/realtime?duplex=1` | OpenAI Realtime-style events (the normative contract) | Applications and browser clients |
| `WS /v1/duplex` | Alias of `/v1/realtime?duplex=1` (same protocol, same handler) | Clients that prefer a dedicated path |
| Python: `DuplexOmni` / `InlineDuplexClient` | Typed commands and events in-process | Embedding the model without a server |

## Enable Full Duplex

A model is served full duplex when its registered pipeline declares a
`duplex_plugin` (the model's `DuplexModelPlugin`). That declaration alone
decides it: `vllm-omni serve` constructs `DuplexOmni` instead of `AsyncOmni`,
and every surface the server exposes is backed by a duplex session. It serves
`/v1/realtime?duplex=1` (and its alias `/v1/duplex`),
`POST /v1/chat/completions`, `/v1/models` and `/health`; every other
turn-based HTTP route (speech, batch, embeddings, video, ...) answers "not
available". Turn-based use of such a model stays available offline through the
Python API (`Omni` / `AsyncOmni`).

`/v1/chat/completions` is served by an adapter that runs one short-lived
duplex session per request. How the prompt gets in depends on what it is, and
the difference is the model's rather than the adapter's:

- **Speech is a turn.** Audio content is appended to the input buffer and
  committed, exactly as a websocket client does it.
- **Text is not.** A model-native model decides to speak from the audio it
  hears, so silence is its signal *not* to take a turn and a text prompt has no
  turn to start. It reaches the model as the session's seeded opening turn
  (`initial_user_text`), and the session is given silence units to generate on.

Only a model that declares `DuplexCapabilities.supports_chat_completions` can
be reached with text; a request to any other is refused with 400 rather than
left waiting out the session idle timeout. A model that should not serve the
route at all lists it in the deploy configuration's `endpoint_restrictions`.

Two consequences of that design are user-visible:

- Every chat request holds an admission slot for its lifetime, so
  `duplex_session.max_sessions` bounds HTTP concurrency as well as websocket
  sessions; beyond it the request is refused with HTTP 503.
- The answer arrives at the model's real-time pace rather than at turn-based
  speed, because a model-native session generates per audio unit.

`n > 1`, `logprobs` and `tools` are refused with HTTP 400: one request is one
duplex turn, and those have no representation in it. So are image and video
content parts -- a Realtime conversation item carries text and audio only, and
answering without the image the caller sent would be worse than refusing.
`chat_template_kwargs.use_tts_template` is honoured (the session config has the
same switch); its other keys are logged as ignored, because a duplex session
renders its own prompt.

The deploy configuration of such a model must agree:

```yaml
session_mode: duplex
```

A duplex model started with a deploy configuration that does not set it fails
at startup rather than falling back to turn-based serving.

!!! warning

    On a deployment that is *not* duplex, `WS /v1/duplex` fails with
    `Duplex API is not available` and `/v1/realtime?duplex=1` falls back to the
    ordinary turn-based realtime handler. Confirm that
    `session.created.session.capabilities` is present before treating the
    connection as full duplex.

**MiniCPM-o 4.5** (`vllm_omni/deploy/minicpmo_4_5.yaml`) is the only model
served over this endpoint today. PersonaPlex and Nemotron VoiceChat still carry
their pre-framework duplex code: their pipelines declare no `duplex_plugin`, so
they run turn-based until the follow-up PRs port them to the plugin contract
(RFC [vllm-omni#7181](https://github.com/vllm-project/vllm-omni/issues/7181)).

JoyVL is a separate HTTP interaction orchestrator and does not use these
WebSocket endpoints. See [Standalone Experimental Servers](standalone_servers.md).

## MiniCPM-o Quick Start

Start the duplex deployment:

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --deploy-config vllm_omni/deploy/minicpmo_4_5.yaml \
  --trust-remote-code \
  --port 8091
```

Stream a mono, PCM16, 16 kHz WAV file with the provided client:

```bash
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
  --url 'ws://localhost:8091/v1/realtime?duplex=1' \
  --model openbmb/MiniCPM-o-4_5 \
  --input-wav input_16k_mono.wav \
  --ref-audio reference_voice.wav \
  --output-dir /tmp/minicpmo-duplex
```

## Realtime Event Lifecycle

A typical `/v1/realtime?duplex=1` session follows this lifecycle:

1. Send `session.update` with the model, modalities, audio formats, and session
   options.
2. Wait for `session.created`; inspect `session.capabilities` instead of
   assuming every model supports the same controls.
3. Send `input_audio_buffer.append` events while microphone audio arrives.
4. Send `input_audio_buffer.commit` at a user-turn boundary when required by
   the model policy.
5. Consume `response.created`, transcript deltas, `response.output_audio.delta`, and
   `response.done` or `response.listen` events.
6. Send `playback.ack` after audio has been played when the session advertises
   playback acknowledgement support.
7. Send `session.close` and wait for `session.closed`.

Unlike the turn-based realtime endpoint, input may continue while a response
is active. The server can emit `overlap.decision` to describe whether input was
deferred, treated as a short acknowledgement, or used to interrupt output.

## Capabilities and Model Differences

The `session.created` payload includes capability fields such as
`supports_barge_in`, `supports_playback_ack`, `supports_multi_session`,
`supports_session_resume`, and `chunk_period_ms`. Treat this payload as the
runtime contract and branch on the flags, never on the model name: a model
that supports native overlapping speech may still advertise
`supports_barge_in=false` when destructive output interruption and model-state
rewind are not validated for it. Capacity and session-resume behavior also
depend on the selected deployment configuration. The per-model table lives in
the [Realtime Duplex API guide](realtime_duplex_api.md#capability-negotiation-by-model).

## Python API

`vllm_omni.entrypoints.duplex_omni.DuplexOmni` runs the same engine-resident
sessions in-process: `open_session()` returns a `DuplexSessionHandle` whose
`submit()` takes typed `DuplexCommand` objects and whose `events()` yields
typed `DuplexEvent` objects (each with a `to_realtime()` wire rendering).
`vllm_omni.clients.inline_duplex.InlineDuplexClient` exposes that handle
behind the `DuplexClient` API. See the
[Realtime Duplex API guide](realtime_duplex_api.md#using-the-python-api).

See the [MiniCPM-o example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/minicpmo)
and the [full-duplex runtime design](../design/fullduplex.md)
for model-specific validation and architecture details.
