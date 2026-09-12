# Shared Realtime protocol extraction

This is the behavior-preserving extraction portion of
[RFC #6592 P0a](https://github.com/vllm-project/vllm-omni/issues/6592), preparing
the shared codec needed by [RFC #7223](https://github.com/vllm-project/vllm-omni/issues/7223).
It does not add Qwen Realtime support or require the runtime redesign in
[#7181](https://github.com/vllm-project/vllm-omni/issues/7181).

## Boundary

`entrypoints/realtime/` owns wire validation, session defaults, conversation
item projection, audio-format conversion, and response event serialization.
`RealtimeSessionState` remains the same wire-only state object; it does not
replace `DuplexSession` or own engine resources.

| Module | Responsibility |
| --- | --- |
| `session.py` | Compose input/output codecs; preserve input and output ordering |
| `input.py` | Decode session, conversation, audio-buffer, and response events into existing internal intents |
| `output.py` | Project internal output into Realtime events, including response/item identity and terminal events |
| `state.py` | Per-connection wire state and response projection records |
| `audio.py` | Audio formats and sample-rate conversion |
| `config.py` | Turn-detection configuration validation and existing native opt-in aliases |
| `runner.py`, `adapters/base.py` | Supply a connection-local codec to a `RealtimeModelAdapter` |

The adapter owns session execution. The existing `OmniDuplexSessionHandler`
implements `RealtimeModelAdapter.handle_session` structurally and remains the
first consumer, using the deployment-selected `ServingRuntimeAdapter` for
MiniCPM or another supported duplex model. Admission, model input submission,
response creation, cancellation, and cleanup still run through its existing
ordered mailbox. The native route `/v1/duplex` is unchanged.

The shared package has no dependency on a duplex handler, engine control plane,
model implementation, or the OpenAI API server. Existing private names such as
`_to_duplex_event` describe the inherited internal event vocabulary; they do not
require a consumer to use the duplex runtime.

## Deliberate limits of this draft boundary

The RFC sketches `entrypoints/openai/realtime/`. The current
`entrypoints/openai/__init__.py` eagerly imports the API server, which imports
duplex serving. Putting a codec used by duplex under that package would create
an import cycle. A sibling `entrypoints/realtime/` keeps the dependency one-way
without changing server initialization or its package exports.

The RFC also sketches per-operation `build_prompt`, `start_response`, `cancel`,
and `supports` methods. This extraction does **not** pretend that native duplex
can be reduced to one `generate()` call per response: model-owned LISTEN/SPEAK
and continuous append remain in its existing execution owner. The draft uses
the session-level adapter entry point actually consumed by that owner. A
turn-based Qwen adapter and any finer-grained shared execution interface remain
P0b work; this is not a claim that the whole RFC or its proposed adapter API is
implemented.

## Compatibility and validation

The old `entrypoints/duplex/realtime_session.py`, `realtime_state.py`, and
`audio.py` paths re-export the same implementations, not copies.
`NativeRealtimeSessionProtocol` is an alias of `RealtimeSessionProtocol`.
Existing duplex tests keep their imports and wire assertions unchanged.
The input/output implementation mixins move without unused compatibility
modules. Server VAD execution stays under duplex; only its pure configuration
parser moves. Audio conversion uses the repository-required `pybase64` API
after extraction.

The extraction deliberately keeps existing defaults, compatibility aliases,
native extensions (including `video_frames`), update barriers, response
identity, and completion ordering. It does not claim blanket OpenAI GA
conformance, add a profile selector, change the legacy Qwen STT route, or add
video handling for Qwen.

Focused CPU coverage includes old-path identity, connection isolation, adapter
failure/cancellation propagation, and an isolated import-boundary check.
The existing duplex protocol, handler, server VAD, and audio suites are the
wire-behavior regression gate. Real MiniCPM and PersonaPlex session tests remain
required before declaring the extraction ready to merge; static checks alone
do not establish runtime compatibility.
