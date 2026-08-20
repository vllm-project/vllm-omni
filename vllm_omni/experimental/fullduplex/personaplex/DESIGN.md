# PersonaPlex Unified Full-Duplex Design

## Status and target

This design adapts PersonaPlex PR #4771 to the unified full-duplex runtime on
top of:

- vLLM-Omni `origin/main`: `67c54777bb22e9e7e08fdf7c47a64f06b566fc47`
- PersonaPlex PR head: `477fb7c225f0c06991bc8aa55eadbd908ba282e4`

The target is the engine-native path:

```text
/v1/realtime?duplex=1
  -> OpenAI Realtime session actor
  -> DuplexRequestClient
  -> AsyncOmni correlated RPC
  -> DuplexControlPlane
  -> resumable Stage 0 request
  -> PersonaPlex Talker
  -> streaming PersonaPlex Code2Wav
  -> PersonaPlex data-plane projector
  -> response.audio.delta + response.audio_transcript.delta
```

The standalone `/api/chat` and `/v1/audio/duplex` server remains a compatibility
surface. It is not evidence that the unified engine path works.

## Why configuration-only enablement is invalid

The current staged pipeline is explicitly turn based. Its Talker reads
`pplex_user_codes`, `pplex_prefill_text`, and `pplex_silence_codes`, but no
production staged input path writes those fields. The voice prompt, persona
prefill, and streaming Mimi state live only in the standalone
`PersonaPlexEngine`.

Setting only the following fields would therefore advertise an endpoint whose
model never receives the live microphone stream:

```python
duplex_control_enabled = True
duplex_runtime_extension = "..."
duplex_serving_adapter = "..."
```

The adapter must supply a real scheduler data plane, not just endpoint
capabilities.

## Supported scope

The unified implementation supports:

- up to two engine-owned sessions on one replica;
- 24 kHz mono float PCM input;
- one 1920-sample, 80 ms model frame per physical append unit;
- continuous user input while assistant audio is generated or played;
- bundled `.pt` voice prompts and a session persona;
- greedy text and depformer sampling, matching the current PersonaPlex port;
- `/v1/realtime?duplex=1`;
- engine lease close, disconnect cleanup, reconnect after cleanup, and explicit
  response cancellation without cross-session state reuse.

The implementation does not claim:

- more than two simultaneous PersonaPlex sessions on one replica;
- arbitrary WAV voice cloning;
- turn-based `response.create` semantics for an otherwise continuous model;
- scheduler migration of live codec state between replicas;
- exact output equality with the standalone engine after different scheduling
  boundaries.

The capability payload derives multi-session support from the configured
session limit. The shipped two-session deployment reports
`supports_multi_session=true` and `supports_multi_session_same_replica=true`.
`duplex_session.max_sessions` is the only capacity source: config resolution
propagates it to every stage as `duplex_max_sessions`, and both Mimi pools read
that model-config value. Connector extras do not carry a second model-specific
capacity knob that could drift from engine admission.
It still reports `supports_barge_in=false`: the generic epoch fence can suppress
stale transport output, but neither PersonaPlex nor the current MiniCPM-o 4.5
adapter proves that model-owned streaming state can be destructively rewound or
restarted at a playback cursor. Continuous overlapping speech is model-native
duplex behavior, not by itself a barge-in contract.

## Components and ownership

### Serving adapter

`PersonaPlexServingRuntimeAdapter` owns only serving-side state:

- a transactional PCM append buffer;
- validation of `voice_prompt` and `instructions`;
- public capabilities;
- PersonaPlex data-plane output projection.

It accepts 24 kHz `pcm_f32le`, groups client packets into whole 1920-sample
frames, zero-pads only the final residual, and rolls a reservation back when an
engine append fails. It never loads CUDA weights and never encodes user audio.

Client input cannot provide local filesystem paths. A voice is a bundled
basename such as `NATF2.pt`; the worker resolves it under the local model
checkpoint. `instructions` is the persona string.

### Runtime extension

`PersonaPlexDuplexRuntimeExtension` is pure model policy. It:

- configures greedy Stage 0 sampling and bounded segment lengths;
- maps each accepted PCM append to a scheduler prompt;
- places immutable session identity, append sequence, PCM payload, voice, and
  persona under `model_intermediate_buffer["duplex"]`;
- reserves exactly one scheduler prompt slot per encoded Mimi frame, plus the
  first-append voice/persona prefill length;
- never performs model inference or owns session state.

The extension returns no turn/listen decision. PersonaPlex is an always-clocked
model, so visible audio/text comes from the final stage data plane.

### Stage 0 streaming runtime

The Talker owns a `PersonaPlexStage0DuplexRuntime` helper, analogous to
MiniCPM-o's Stage 0 helper but with PersonaPlex lockstep semantics.

For each admitted session it owns:

- streaming Mimi encoder convolution and transformer state;
- the selected voice embedding bundle;
- persona tokenization and prefill embeddings;
- the prior user code frame needed by the one-frame acoustic delay;
- append identity used to make a retried scheduler update idempotent.

The first append builds this ordered prefill:

```text
voice embeddings
  -> encoded silence
  -> <system> persona <system> tokens with encoded silence
  -> encoded silence
  -> first live user frame
```

Later appends encode only new 1920-sample frames. The helper returns the
per-frame user codes and prompt embeddings through the request's
`model_intermediate_buffer`. The normal vLLM runner remains authoritative for
attention metadata, block tables, KV allocation, scheduling, and sampling.

The Talker must distinguish resumable prompt prefill from decode positions.
Prompt rows consume the exact prepared embeddings; sampled decode rows continue
to use the existing delayed agent/user frame construction. No code path may
fall back to an all-initial user stream for a duplex request.

Cleanup is keyed by the full `(session_id, incarnation)` identity. Every live
session has an independent Mimi encoder instance; encoder convolution/KV state
is never shared between asynchronously scheduled sessions. A finished or
aborted scheduler request resets and returns only that session's encoder.

### Stage 1 streaming decoder

The current `PersonaPlexCode2Wav` calls one-shot `MimiModel.decode` and is not
CUDA-graph safe. Unified duplex uses eager Stage 1 and maintains an independent
streaming Mimi decoder for every active request. Decoder ownership is keyed by
the stable Stage 1 request id and released by `on_requests_finished`; a mixed
batch must never advance another request's convolution or transformer state.

Each Stage 0 segment emits de-delayed agent codebooks. Stage 1 decodes only the
new code frames, emits only the new PCM suffix, and resets state when the
request is closed. Connector chunk boundaries retain the final raw code frame
needed to de-delay the next chunk.

The deploy default sets Stage 1 `enforce_eager: true`; a default configuration
that fails during CUDA graph capture is not an acceptable deployment profile.

### Data-plane projector

`PersonaPlexDataPlaneSession` converts cumulative or delta Stage 1 output into
model-neutral native results:

```python
{
    "stage_role": "tts",
    "data_plane_request_id": request_id,
    "text": text_delta,
    "audio_data": encoded_audio_delta,
    "audio_format": response_format,
    "sample_rate_hz": 24000,
    "audio_duration_ms": delta_duration,
    "end_of_turn": False,
}
```

It owns per-request audio and text cursors so a cumulative output cannot replay
old audio. PersonaPlex keeps one visible response open while continuous output
arrives. Session close or cancellation terminates that response through the
generic Realtime lifecycle; a codec segment finishing is not a conversational
turn boundary.

## Error and lifecycle contracts

- Unsupported sample rate, malformed base64, non-finite PCM, invalid voice
  basename, or changed format fails before scheduler submission.
- Append is prepare/submit/commit. Failure rolls back the exact reserved PCM
  bytes and does not advance the model frame cursor.
- A repeated `operation_id` must not encode or submit the same frame twice.
- Input iterator exceptions execute the same cleanup as explicit close.
- Cancellation does not release the engine lease until the stage request and
  any in-flight codec operation are actually finished.
- A bounded cleanup timeout returns a cleanup error and keeps the session in
  the closing admission set; it must not make the slot available while work
  still mutates shared state.
- New sessions cannot observe the previous voice, persona, PCM tail, Mimi
  convolution state, or Talker delayed code frame.

The legacy `PersonaPlexDuplexRuntime.run()` and standalone server drain path
must follow the same exception-safe rule while they remain in the tree.

## Testing and acceptance

### Contract tests

Tests first cover:

- pipeline registration enables the control plane and selects both PersonaPlex
  adapters;
- the capability payload is two-session, 80 ms, append-only, and honest;
- PCM reservation commit/rollback, partial-frame flush, invalid input, and
  operation idempotency;
- runtime prompt fields and exact token budgeting;
- first-append voice/persona prefill followed by live user codes;
- later appends retain per-session Mimi state and do not replay prefill;
- interleaved Stage 0 and Stage 1 work preserves independent codec histories;
- output projection emits only audio/text deltas;
- close, exception, timeout, and reconnect cleanup;
- ordinary non-duplex imports do not load PersonaPlex modules.

### Remote H20 validation

Validation runs in an isolated remote worktree using the ModelScope
`nv-community/personaplex-7b-v1` checkpoint and its Mimi dependency.

The required evidence is:

1. default `personaplex.yaml` reaches ready without a local eager override;
2. `/health` returns 200;
3. `/v1/realtime?duplex=1` reports `model_native_duplex`, `chunk_period_ms=80`,
   and two-session admission;
4. paced 24 kHz PCM appends produce finite, non-silent 24 kHz audio deltas and
   text deltas;
5. microphone input continues during assistant output without cancelling the
   scheduler request;
6. two paced sessions simultaneously produce independent non-empty audio, and
   a third session is rejected with `resource_exhausted`;
7. closing either session frees only its scheduler request and GPU codec state,
   after which a replacement session can use a different persona without state
   leakage;
8. malformed input returns one typed error and does not poison the next append;
9. all focused unit tests and `git diff --check` pass.

Audio that is empty, all zero, non-finite, or only a protocol `listen` event is
not a successful end-to-end result.
