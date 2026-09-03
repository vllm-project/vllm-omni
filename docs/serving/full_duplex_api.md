# Full-Duplex WebSocket API

vLLM-Omni provides an experimental full-duplex runtime for models that can
continue receiving speech while producing speech. It adds persistent session
state, model-specific turn policy, overlap handling, playback acknowledgement,
and optional session resume.

Full duplex is distinct from the turn-based [Realtime Audio API](realtime_api.md).

## Choose an Endpoint

| Endpoint | Protocol | Recommended use |
|----------|----------|-----------------|
| `WS /v1/realtime?duplex=1` | OpenAI Realtime-style event projection | Applications and browser clients |
| `WS /v1/duplex` | Native vLLM-Omni duplex events | Runtime integration and low-level testing |

Both endpoints use the same duplex engine and require the same model-side
adapter. Prefer the Realtime projection unless the native lifecycle events are
specifically required.

## Enable Full Duplex

The route becomes usable only when the deployment configuration explicitly
sets:

```yaml
session_mode: duplex
```

The selected model pipeline must also provide a duplex serving adapter.
Model-native deployments configure the corresponding engine runtime extension
and control plane as part of their registered pipeline.

!!! warning

    `WS /v1/duplex` fails with `Duplex API is not available` when duplex is not
    enabled. By contrast, `/v1/realtime?duplex=1` falls back to the ordinary
    turn-based realtime handler when no duplex handler exists. Confirm that
    `session.created.session.capabilities` is present before treating the
    connection as full duplex.

The current unified-runtime integrations are:

- MiniCPM-o 4.5, using `vllm_omni/deploy/minicpmo_4_5_duplex.yaml`;
- PersonaPlex, whose default `vllm_omni/deploy/personaplex.yaml` enables duplex.

JoyVL is a separate HTTP interaction orchestrator and does not use these
WebSocket endpoints. See [Standalone Experimental Servers](standalone_servers.md).

## MiniCPM-o Quick Start

Start the duplex deployment:

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --deploy-config vllm_omni/deploy/minicpmo_4_5_duplex.yaml \
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
5. Consume `response.created`, transcript deltas, `response.audio.delta`, and
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
runtime contract.

For example, PersonaPlex supports native overlapping speech but currently
advertises `supports_barge_in=false`; destructive output interruption and
model-state rewind have not been validated for that integration. Capacity and
session-resume behavior also depend on the selected deployment configuration.

## Native Protocol

`WS /v1/duplex` exposes lower-level lifecycle names including
`session.create`, `input_audio_buffer.append`, `turn.signal`, `playback.ack`,
and `session.close`. It returns native session, input, response, overlap, and
error events. This protocol is experimental and may evolve with the runtime;
applications should use the provided Realtime client where possible.

See the [MiniCPM-o example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/minicpmo),
[PersonaPlex example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/personaplex),
and [full-duplex runtime design](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/experimental/fullduplex/DESIGN.md)
for model-specific validation and architecture details.
