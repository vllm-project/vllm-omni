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

- MiniCPM-o 4.5, using `vllm_omni/deploy/minicpmo_4_5.yaml`
  (`session_mode: duplex`);
- PersonaPlex, whose default `vllm_omni/deploy/personaplex.yaml` enables duplex;
- Nemotron VoiceChat, via its registered duplex plugin package;
- [Gander Unit8 dialogue and tool/context inputs](gander.md), reusing the MiniCPM-o 4.5 pipeline
  with `vllm_omni/deploy/gander.yaml`.

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

For `response_lifecycle=continuous_stream` (Nemotron), a model EOS only ends
an utterance transcript. Keep supplying audio, including trailing silence
for file input, then send `session.close`. Do not wait for `response.done`
before closing: normal close first drains accepted text/audio frames, emits
`completed` with reason `stream_drained`, and then sends `session.end` and
`session.closed`. Explicit cancel remains a cancellation. See the
[official alignment report](../validation/nemotron_official_alignment_20260907.md).

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

## Connection recovery and retained state

When `supports_session_resume` is advertised, a replacement connection sends
`session.resume` with `session_id`, `incarnation`, the last delivered
`resume_token`, and `last_received_server_event_seq`. The server serializes
lease changes for that session, revokes the old attachment, sends
`session.resumed`, and replays retained events before sending new live events.
If activation is cancelled or its delivery fails, the last accepted credential
remains usable for recovery while detached; a successful recovery revokes it.
The failed handshake also restores the engine's detached lease state before a
retry, so disconnect grace still applies. Cancelling before takeover does not
detach an old connection that is still live.
An expired session cannot be restored this way.

The deploy-only setting `duplex_session.attachment_io_timeout_s` defaults to
`5.0` seconds and must be finite and positive. It bounds the whole resume
activation/replay phase, and each terminal notification/close attempt; clients
cannot override it through session payloads. Ordinary live output preserves
transport backpressure, but a takeover or close can revoke its blocked send.
Terminal notifications are best-effort: an expired session's resource cleanup
does not wait for notification delivery, and one session's cleanup does not
serialize another's. A temporary disconnection still uses `disconnect_grace_s`;
it does not immediately destroy the retained runtime session or KV.

The replay journal stores immutable serialized event snapshots. Its byte budget
is computed from those snapshots, so later transcript updates cannot change an
earlier event or its retained-byte accounting.

The standard WebSocket transport sends these encoded snapshots as text frames
without decoding and encoding them again. Dictionary-only transport callbacks
remain supported. On Python 3.12+, nonblocking sends can complete in an isolated
eager task when the event loop has no custom task factory; other environments
use the scheduled path. Inline bursts yield every 32 sends so control tasks can
still run. Blocking sends retain cancellation/revocation and ordering guarantees.

On response completion, the Realtime projector releases its text/transcript
assembly state and keeps at most 256 recent terminal IDs. On the serving path,
only `response.created` admits a response; late deltas cannot recreate a retired
response, even after its terminal ID leaves this cache. The completed
conversation item remains the owner of history and audio-to-text truncation
marks until the item is deleted or the session ends. This cleanup does not
silently evict conversation history, summarize context, or compact model KV.

## Native Protocol

`WS /v1/duplex` exposes lower-level lifecycle names including
`session.create`, `input_audio_buffer.append`, `turn.signal`, `playback.ack`,
and `session.close`. It returns native session, input, response, overlap, and
error events. This protocol is experimental and may evolve with the runtime;
applications should use the provided Realtime client where possible.

See the [MiniCPM-o example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/minicpmo),
[PersonaPlex example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/personaplex),
and the [full-duplex runtime design](../design/fullduplex.md)
for model-specific validation and architecture details.
