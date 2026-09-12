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
  -> response.output_audio.delta + response.output_audio_transcript.delta
```

The standalone `/api/chat` and `/v1/audio/duplex` server that accompanied the
original PR was demo-only and has been removed from the tree; the unified
engine path above is the only serving surface. It was never evidence that the
unified engine path works.

## Why configuration-only enablement was invalid

The staged pipeline as shipped by the original PR was explicitly turn based.
Its Talker read `pplex_user_codes`, `pplex_prefill_text`, and
`pplex_silence_codes`, but no production staged input path wrote those fields.
The voice prompt, persona prefill, and streaming Mimi state lived only in the
standalone `PersonaPlexEngine` (demo-only, since removed).

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
- `/v1/realtime?duplex=1` (the wire vocabulary is catalogued in the
  [Realtime Duplex API](../serving/realtime_duplex_api.md) serving guide);
- the public client preset
  `vllm_omni.clients.personaplex.create_duplex_session_config()` (24 kHz
  `pcm_f32le` input format, voice prompt, persona) as the canonical
  session-config source for `DuplexClient` consumers;
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
old audio. The generic projector turns each native result into the Realtime
`response.output_audio.delta` / `response.output_audio_transcript.delta` pair (and
`response.output_text.delta` for text) under one `response_id`; the full
mapping is the name map in the
[Realtime Duplex API](../serving/realtime_duplex_api.md) serving guide.
PersonaPlex keeps one visible response open while continuous output arrives.
Session close or cancellation terminates that response through the generic
Realtime lifecycle; a codec segment finishing is not a conversational turn
boundary. PersonaPlex advertises `supports_barge_in=false` and
`supports_session_resume=false`, so `barge_in`, `turn_detection.server_vad`,
and `session.resume` are rejected on this model.

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

(The legacy `PersonaPlexDuplexRuntime.run()` and standalone server drain path,
which followed the same exception-safe rule, have been removed from the tree.)

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

### Paced multi-session load driver

The existing driver keeps its two-session admission and slot-reuse validation
when `--sessions` is omitted. Passing `--sessions N` selects a separate,
bounded load mode; it does not replace the lifecycle test or change server
admission capacity. Start an existing PersonaPlex server configured to admit
the requested number of sessions, then run from a source checkout with the
vLLM-Omni client dependencies installed:

```bash
python -m tests.e2e.online_serving.personaplex_realtime_duplex \
  --model nvidia/personaplex-7b-v1 \
  --input-wav /path/to/input.wav \
  --sessions 2 --load-frames 1000 --drain-s 2 \
  --server-revision SERVER_COMMIT --server-hardware A100-80GB \
  --output-dir /tmp/personaplex-load-n2
```

Use a new output directory for every run. The driver refuses to overwrite an
existing `load-result.json`. The server labels are supplied by the operator,
not detected or verified by the client. Record the server command, package
versions, checkpoint revision and hardware alongside a published measurement.

All attempted sessions finish admission before admitted sessions share a
common input start. A rejected session remains a failed row in the report;
it is not removed from the denominator. Each session receives at most
`--load-frames` input frames, without looping the WAV, and uses 80 ms pacing.
A stalled send cannot cause a catch-up burst. The timeline records send
schedule lateness so an overloaded client is visible rather than mistaken
for a faster server. This mode measures steady concurrency, not join/leave
interference; the original lifecycle mode still checks slot reuse.

The fixed drain window is part of the measurement configuration. Audio
arriving during cleanup is excluded, so closing cannot rescue a session that
missed its observation deadline. Cleanup is attempted for admitted sessions
even when audio validation fails. Protocol errors, missing/invalid audio,
shared response IDs and cleanup failures make the report fail and the CLI
exit nonzero. Cancellation closes transports and propagates to the caller.

`load-result.json` retains per-session send and audio-packet timelines using
the client's monotonic clock. It does not export handshake credentials or
raw audio payloads. Metric definitions are explicit:

Load mode validates each nonempty audio packet's strict base64 encoding, PCM16
sample alignment, 24 kHz sample rate and response identity. Valid aggregate
audio cannot hide a malformed packet. Rejected packets make the session fail
and appear in `invalid_audio_packets` with their event index, receive time and
fixed diagnostic code; they do not contribute samples or timing intervals.
Empty audio deltas remain legal and contribute no samples. This check does
not change the public client's decoding or the default lifecycle mode.

The probe accepts both the current OpenAI Realtime audio event name
`response.audio.delta` and the legacy `response.output_audio.delta`. Validation
uses the wire payload directly rather than relying on a particular client
library's alias table, so a server-side event-name migration cannot silently
turn real audio into a zero-output measurement.

| Field | Meaning |
| --- | --- |
| `client_audio_packet_interval_ms` | Consecutive nonempty packet arrivals: median, linear p99, max, and zero-based index of the first maximal interval. Interval index 0 ends at packet 1. One packet can contain multiple codec frames; this is not server tick latency. |
| `client_first_audio_after_stream_start_ms` | First audio receipt minus first input send start; admission is excluded. |
| `client_stream_rtf` | First input send start to last audio receipt, divided by received audio duration. This includes real-time input pacing and is not an inference-only RTF. |
| `client_send_lateness_ms` | Send start minus its original shared schedule, clipped at zero. |
| `frame_deficit` | Sent input frames minus received samples divided by 1920. |

No output gives missing latency/RTF values, not zero. By default RTF is
diagnostic; `--max-client-rtf` adds an explicit ceiling. The existing frame
coverage, audible-frame and scheduler-data-plane checks still apply. A longer
`--drain-s` can demonstrate eventual completion under backlog, but it must not
be reported as real-time capacity when the fixed-window run failed or client
RTF exceeded the stated target. Likewise, the first live session can pay
one-time model/codec/JIT costs; either use the server's configured duplex
warmup path or label that run as cold-start evidence before comparing steady
capacity. Passing client-only metrics does not certify model quality or a
real-time capacity target on another GPU. Begin with N=1 and N=2; higher N is a
capacity experiment, not a promised property of an A100 or of this patch.

Deterministic driver and localhost WebSocket checks need no model or GPU:

```bash
python -m pytest tests/e2e/online_serving/test_personaplex_load_driver.py \
  -m 'core_model and cpu' --run-level core_model -q
```

These checks test the driver against controlled protocol responses. Run the
load-driver command against real weights separately before reporting model-serving
performance. Server tick timestamps and generic TTFT/TPOT aggregation remain
the responsibility of the existing duplex metrics paths, not this driver.
