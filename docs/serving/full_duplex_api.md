# Full-Duplex WebSocket API

vLLM-Omni provides a full-duplex runtime for models that can
continue receiving speech while producing speech. It adds persistent session
state, model-specific turn policy, overlap handling, playback acknowledgement,
and optional session resume.

Full duplex is distinct from the turn-based [Realtime Audio API](realtime_openai_duplex_api.md).

This page is the endpoint overview. For the complete wire contract, the
`vllm_omni.clients.duplex.DuplexClient` Python library, and the per-model
capability gates, see the [Realtime Duplex API guide](realtime_duplex_api.md).

## Run the duplex runtime in a separate process

The default `DuplexOmni` backend runs the orchestrator on a background thread
in the frontend process. Use `--duplex-engine-backend-as-process` to give the
native runtime its own Python process and GIL:

```bash
vllm serve "$MODEL" --omni \
  --deploy-config "$DEPLOY_CONFIG" \
  --duplex-engine-backend-as-process
```

Set `MODEL` to the checkpoint and `DEPLOY_CONFIG` to its deployment YAML.
The deployment must resolve to `session_mode: duplex`. The option is independent
of the model plugin, stage count, and stage worker backend.

Enabling this option makes the deployment **session-only**, including when the
model also supports chat. It serves the duplex WebSocket routes, `/v1/models`,
`/health`, and metrics. Chat and other generation routes are unavailable.
The model's reported `DuplexCapabilities` remain unchanged. Keep the default
`duplex_engine_backend_as_process=False` to serve chat alongside sessions.

For embedded Python use, pass `duplex_engine_backend_as_process=True` to
`DuplexOmni`. Protect the program entrypoint for Python's `spawn` start method:

```python
import asyncio
import os

from vllm_omni.entrypoints.duplex_omni import DuplexOmni


async def main():
    omni = DuplexOmni(
        model=os.environ["MODEL"],
        deploy_config=os.environ["DEPLOY_CONFIG"],
        duplex_engine_backend_as_process=True,
    )
    try:
        async with await omni.open_session() as session:
            # Use the same session.submit(), session.events(), and resume APIs.
            await session.heartbeat()
    finally:
        await omni.aclose()


if __name__ == "__main__":
    asyncio.run(main())
```

Process mode currently requires a POSIX host and an importable program
entrypoint. Model registrations must be available in the child through the
normal `vllm.general_plugins` mechanism; parent-only registry mutations are
not inherited. Startup options must be compatible with `spawn`, and session
commands/configuration must contain MessagePack-compatible data. The frontend
receives stage IDs, types, and effective context limits, not full stage
configurations or processors. Generation, tokenization, profiling, sleep/wake,
and weight-transfer administration are outside this session-only interface.

`DuplexOmni` retains session handles and lifecycle behavior;
`ProcessDuplexEngineClient` owns spawning, IPC, and child teardown. The child
owns the native orchestrator thread and stage clients. The processes exchange
MessagePack data over a private socketpair using asyncio, without reader or
writer threads. Both backends share the frontend output pump and engine-death
handling.

Each process keeps bounded per-session output buffers. Each IPC direction
admits at most 2048 queued messages and 16 MiB of encoded queued data; larger
individual messages are rejected. The active write and socket buffers are
additional to this budget. Frontend output overflow closes the affected session;
worker or transport failure fails pending controls and closes local buffers.
`await omni.aclose()` requests graceful worker cleanup; synchronous `shutdown()`
provides bounded process-group termination. Serialization and buffering add
costs, so benchmark the target workload.

Invalidations share the event stream: the frontend removes matching queued
audio and rejects stale epochs, but cannot revoke audio committed while the
notification was in transit. Resume leases and replay control IDs remain owned
by the native engine; process mode adds no cross-frontend session ownership or
resume support.

## Choose an Endpoint

| Endpoint | Protocol | Recommended use |
| --- | --- | --- |
| `WS /v1/realtime?duplex=1` | OpenAI Realtime-style events (the normative contract) | Applications and browser clients |
| `WS /v1/duplex` | Alias of `/v1/realtime?duplex=1` (same protocol, same handler) | Clients that prefer a dedicated path |
| Python: `DuplexOmni` / `InlineDuplexClient` | Typed commands and events through local session handles | Embedding the model without a server |

## Enable Full Duplex

A model is served full duplex when its registered pipeline declares a
`duplex_plugin` (the model's `DuplexModelPlugin`) and its deploy configuration
sets `session_mode: duplex`. `vllm serve --omni` constructs `DuplexOmni`
instead of `AsyncOmni`. It serves
`/v1/realtime?duplex=1` (and its alias `/v1/duplex`),
`POST /v1/chat/completions`, `/v1/models` and `/health`; every other
turn-based HTTP route (speech, batch, embeddings, video, ...) answers "not
available". Set `session_mode: turn` to select the ordinary online serving
stack instead. The model's supported tasks and endpoint restrictions still apply.

`/v1/chat/completions` is the ordinary chat service running on the duplex
engine: a request is a turn-based generation on the same stages, served
alongside the live websocket sessions, with the request options the chat
service supports. It is wired only when the model's plugin declares
`DuplexCapabilities.supports_chat_completions`; on any other duplex model the
route reports "not available". A model that should not serve the route at all
lists it in the deploy configuration's `endpoint_restrictions`.

The deploy configuration of such a model must agree:

```yaml
session_mode: duplex
```

A duplex-capable model must explicitly set `session_mode` to `duplex` or `turn`
in its deploy configuration, possibly through `base_config` inheritance.
Missing or invalid values fail at startup. The default MiniCPM-o deployment
continues to use duplex mode.

To run MiniCPM-o 4.5 with the turn-based engine:

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
  --deploy-config vllm_omni/deploy/minicpmo_4_5_turn.yaml \
  --trust-remote-code \
  --port 8091
```

This profile inherits the default model and stage settings and overrides only
`session_mode`. It supports ordinary HTTP requests without creating a duplex
session handler. Mode selection happens at startup; a WebSocket query parameter
does not switch engines. The Python `Omni` / `AsyncOmni` APIs are unchanged.

!!! warning

    On a deployment that is *not* duplex, `WS /v1/duplex` fails with
    `Duplex API is not available` and `/v1/realtime?duplex=1` falls back to the
    ordinary turn-based realtime handler. Confirm that
    `session.created.session.capabilities` is present before treating the
    connection as full duplex.

**MiniCPM-o 4.5** (`vllm_omni/deploy/minicpmo_4_5.yaml`), **Qwen3-Omni**
(`vllm_omni/deploy/qwen3_omni_duplex.yaml`), and **Nemotron VoiceChat**
(`vllm_omni/deploy/nemotron_labs_voicechat_duplex.yaml`) are served
over this endpoint today. CUDA Qwen3-Omni deployments can use
`vllm_omni/deploy/qwen3_omni_duplex_mrv2.yaml` to run the same duplex session
on Model Runner V2; non-CUDA platforms in that profile stay on V1. PersonaPlex
carries its pre-framework duplex code: its pipeline declares no `duplex_plugin`,
so it runs turn-based until the
follow-up PR ports it to the plugin contract
(RFC [vllm-omni#7181](https://github.com/vllm-project/vllm-omni/issues/7181)).

For Nemotron VoiceChat Python clients, use
`vllm_omni.clients.nemotron_voicechat.create_duplex_session_config()` with
`DuplexClient`. The preset selects 16 kHz float32 input, 22.05 kHz PCM16 output,
and automatic responses. Append 1,280-sample (80 ms) input frames; when using
`stream_pcm`, set `chunk_ms=80`. Pass `instructions` and optional `tools` to
the preset before opening the session.

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
