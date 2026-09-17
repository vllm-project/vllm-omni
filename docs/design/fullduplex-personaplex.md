# PersonaPlex on the Unified Full-Duplex Framework

## Status and target

PersonaPlex (`nvidia/personaplex-7b-v1`, a Moshi finetune) is served by the
[Unified Full-Duplex Framework](fullduplex.md) through one model plugin,
`PersonaPlexDuplexPlugin` (`vllm_omni/model_executor/models/personaplex/duplex/plugin.py`),
selected by `PipelineConfig.duplex_plugin`. The serving path is:

```text
/v1/realtime?duplex=1
  -> OmniDuplexSessionHandler (thin: websocket I/O, Realtime <-> DuplexCommand/DuplexEvent)
  -> DuplexOmni / DuplexOmniEngine
  -> DuplexSessionManager -> DuplexSessionRunner (one per session)
       plugin.plan_append: one 80 ms frame -> one resumable Stage 0 append
  -> PersonaPlex Talker (Stage 0, lockstep temporal transformer + depformer)
  -> streaming PersonaPlex Code2Wav (Stage 1, Mimi)
  -> plugin.data_plane: cumulative audio/text -> deltas
  -> response.output_audio.delta + response.output_audio_transcript.delta
```

The pre-framework pair (engine runtime extension plus serving adapter) and
the standalone `/api/chat` / `/v1/audio/duplex` server are gone; the plugin is
the only integration surface, and this document describes it.

## Supported scope

The integration supports:

- up to `duplex_session.max_sessions` engine-owned sessions on one replica
  (the shipped deploy sets two);
- 24 kHz mono float PCM input, one 1920-sample (80 ms) model frame per
  physical append unit; clients may send any chunking, the session buffers
  whole frames;
- continuous user input while assistant audio is generated or played
  (pure lockstep: the model listens while it speaks);
- bundled `.pt` voice prompts (`voice`) and a session persona (`instructions`);
- greedy text and depformer sampling (one temporal token per frame);
- the public client preset
  `vllm_omni.clients.personaplex.create_duplex_session_config()` (24 kHz
  `pcm_f32le` input, voice, persona) as the canonical session-config source
  for `DuplexClient` / `InlineDuplexClient` consumers;
- engine lease close, disconnect cleanup and explicit response cancellation
  without cross-session state reuse.

It does not claim:

- arbitrary WAV voice cloning (voices are bundled basenames, resolved by the
  worker under the checkpoint);
- turn-based `response.create` semantics, client commits or external turn
  signals (`supports_client_commit=false`, `supports_external_turn_signal=false`);
- text seeding, hence no `/v1/chat/completions` route
  (`supports_chat_completions=false`);
- session resume across a transport drop (`supports_session_resume=false`);
- destructive output interruption or model-state rewind at a playback cursor
  (`supports_barge_in=false`, `supports_audio_truncate=false`): overlapping
  speech is native model behaviour, not a barge-in contract.

`duplex_session.max_sessions` is the only capacity source: config resolution
propagates it to every stage as `duplex_max_sessions`, and both Mimi pools
(Stage 0 encoders, Stage 1 decoders) read that model-config value. The
capability payload derives `supports_multi_session` /
`supports_multi_session_same_replica` from it.

## Components and ownership

### The plugin (engine side)

`PersonaPlexDuplexPlugin` owns both halves of the contract.

Engine policy:

- `configure_sampling_params`: Stage 0 greedy (`temperature=0`, `top_k=1`,
  `max_tokens=1`), other stages untouched.
- `plan_append`: validates exactly one 1920-sample 24 kHz `pcm_f32le` frame
  (through `model_executor/common/duplex/payload.py`) and reserves one
  scheduler slot per frame plus, on the first append of an epoch (`seq == 1`),
  the `personaplex_prefill_slots` of the voice/persona prefill. The prompt
  carries the fence, `seq`, the payload and the runtime config under
  `model_intermediate_buffer["duplex"]`.
- `decide_output`: never decides. PersonaPlex is always-clocked; what the
  client hears comes from the final-stage data plane.
- `silence_unit_payload`: one frame of zeros at 24 kHz
  (`silence_continuation_samples=1920`,
  `silence_continuation_sample_rate_hz=24000`). The runner uses it to keep a
  model turn clocked when the client pauses; the startup warmup sends it.

Session policy:

- `capabilities`: `personaplex_capabilities(max_sessions)` (80 ms units,
  append-only, no commits, no barge-in, no resume, no chat route).
- `prepare_runtime_config`: `voice` must be a bundled `.pt` basename,
  `instructions` defaults to the shipped persona; the prefill slot count is
  computed once per `(model, voice, persona)` off the orchestrator loop
  (`personaplex_prefill_slots`, itself caching the tokenizer and the voice
  bundle row count). The result is server-owned runtime config
  (`personaplex_model_path`, `personaplex_voice_prompt`,
  `personaplex_persona`, `personaplex_prefill_slots`); the same keys are
  refused in a client's `extra_body`.
- `runtime_config_for_update`: persona and voice are immutable for the
  session (`persona_update_unsupported`, `voice_update_unsupported`).
- session state, extra-body validation and data-plane context are the
  framework defaults (`DefaultDuplexModelSessionState` with the 80 ms
  `PersonaPlexPcmAppendBuffer`, `DuplexDataPlaneContext`).

Because `supports_client_commit` is off, the session auto-responds without
`extra_body.auto_response`: a stock Realtime client streams audio and hears
the model without any vendor flag.

### Input framing

`PersonaPlexPcmAppendBuffer` is the `FixedFramePcmAppendBuffer` of
`model_executor/common/duplex/pcm_buffer.py` at 24 kHz / 1920 samples /
80 ms. It accepts 24 kHz `pcm_f32le` only, groups client packets into whole
frames, takes one frame out per append as a reservation (committed when the
stage accepted the append, rolled back to the front of the buffer when it did
not), zero-pads only a final residual on commit, and never encodes audio.

### Stage 0 streaming runtime (worker side)

The Talker owns a `PersonaPlexStage0DuplexRuntime`
(`model_executor/models/personaplex/duplex/stage0.py`), created lazily in
`_duplex_stage0_runtime()` and released through `on_requests_finished`.

For each live `(session_id, epoch)` it owns:

- the streaming Mimi encoder convolution and transformer state;
- the selected voice embedding bundle and the persona prefill embeddings;
- the prior user code frames needed by the one- and two-frame acoustic
  delays;
- the append identity `(epoch, seq)` that makes a retried scheduler update
  idempotent.

The first append of an epoch builds this ordered prefill:

```text
voice embeddings
  -> encoded silence
  -> <system> persona <system> tokens with encoded silence
  -> encoded silence
  -> first live user frame
```

Later appends encode only new 1920-sample frames. The runtime returns the
per-frame user codes and prompt embeddings through the request's
`model_intermediate_buffer`; the normal vLLM runner remains authoritative for
attention metadata, block tables, KV allocation, scheduling and sampling.

**Epochs.** The framework identifies a Stage 0 request by
`(session_id, epoch)`; `response.cancel` and `output_audio_buffer.clear`
advance the epoch, abort the current request and start the next append at
`seq == 1` on a fresh request with fresh KV. The Stage 0 runtime therefore
keys its state by `(session_id, epoch)`: when an append of a newer epoch
arrives, any older-epoch state of the same session is closed first (its Mimi
encoder returns to the pool, so the codec budget never counts a superseded
epoch), and the voice/persona prefill is replayed because the plan reserved
the slots again. The user-visible consequence is that a cancel restarts the
model's conversation context. A late `on_requests_finished` for the aborted
request finds nothing to close.

Every live session has an independent Mimi encoder instance; encoder state is
never shared between asynchronously scheduled sessions. A finished or
aborted scheduler request resets and returns only that session's encoder.

### Stage 1 streaming decoder

`PersonaPlexCode2Wav` runs eager (`enforce_eager: true` in the deploy) and
maintains an independent streaming Mimi decoder for every active request,
keyed by the Stage 1 request id and released by `on_requests_finished`. Each
Stage 0 segment emits de-delayed agent codebooks; Stage 1 decodes only the new
code frames and emits only the new PCM suffix. Connector chunk boundaries
retain the final raw code frame needed to de-delay the next chunk.

### Data-plane projector

`PersonaPlexDataPlaneSession` is the `CumulativeAudioTextDataPlane` of
`model_executor/common/duplex/data_plane.py` with a 24 kHz default rate. It
keeps one audio/text cursor per request so cumulative Stage 1 output cannot
replay old audio, and yields one internal result per new suffix:

```python
{
    "stage_role": "tts",
    "is_listen": False,
    "data_plane_request_id": request_id,
    "text": text_delta,
    "audio_data": encoded_audio_delta,
    "audio_format": response_format,
    "sample_rate_hz": 24000,
    "audio_duration_ms": delta_duration,
    "end_of_turn": False,
    "runtime_impl": "scheduler_data_plane",
    ...
}
```

The runner turns each result into the Realtime
`response.output_audio.delta` / `response.output_audio_transcript.delta` pair
under one `response_id`. PersonaPlex keeps one visible response open while
continuous output arrives; session close or cancellation terminates it through
the generic Realtime lifecycle. A codec segment finishing is not a
conversational turn boundary.

## Error and lifecycle contracts

- Unsupported sample rate, malformed base64, non-finite PCM, a partial frame
  at the plan, an invalid voice basename or a changed format fails before
  scheduler submission, as a typed `error` event.
- Append is prepare/submit/commit. Failure rolls back the exact reserved PCM
  bytes and does not advance the model frame cursor.
- A repeated `(epoch, seq)` identity does not encode or submit the same frame
  twice.
- Cancellation advances the epoch; the aborted request's Stage 0 state is
  released by the next-epoch append or by `on_requests_finished`, whichever
  comes first.
- Close releases the admission slot only once stage cleanup succeeded; the
  reaper retries a failed cleanup while the slot stays held.
- New sessions cannot observe a previous voice, persona, PCM tail, Mimi
  convolution state or Talker delayed code frame.

## Testing and acceptance

### CPU contract tests

- `tests/model_executor/models/personaplex/duplex/test_plugin.py`: pipeline
  binding and plugin load, honest capabilities, private keys, voice/persona
  resolution and caching, immutable updates, greedy sampling, one-slot-plus-
  prefill planning, frame validation, the 24 kHz silence unit, and the
  CPU-checked helpers of the E2E driver.
- `tests/model_executor/models/personaplex/duplex/test_stage0_runtime.py`:
  first-append prefill, depformer teacher forcing, causal user-frame delays,
  idempotent retries, independent encoders per session, capacity, epoch
  restart and late finish of a superseded request.
- `tests/engine/duplex/test_session_runner_personaplex.py`: the session
  runner with the real plugin -- one submission per frame, prefill on the
  first append only, half frames buffered, wrong rate refused, cumulative
  Code2Wav output projected as 24 kHz deltas, cancel restarting the epoch,
  close aborting the request.
- `tests/model_executor/common/`: the shared toolbox (PCM helpers, payload
  validation, fixed-frame buffer, cumulative data plane, request-output
  readers).

### GPU validation

`tests/e2e/online_serving/personaplex_realtime_duplex.py` (wrapped by
`tests/e2e/online_serving/test_personaplex_duplex.py` when
`PERSONAPLEX_MODEL_PATH` is set) drives the default `personaplex.yaml`
deployment and requires:

1. the server reaches ready without a local eager override; `/health` is 200;
2. `/v1/realtime?duplex=1` reports `model_native_duplex`, `chunk_period_ms=80`
   and two-session admission, with server-allocated session ids;
3. paced 24 kHz PCM appends produce finite, non-silent, whole-frame 24 kHz
   audio deltas and text deltas;
4. two paced sessions simultaneously produce independent non-empty audio, and
   a third session is refused with `resource_exhausted`;
5. closing one session frees only its scheduler request and codec state, after
   which a replacement session with a different persona is admitted with a
   fresh id and no state leakage.

Audio that is empty, all zero, non-finite, or only a protocol `listen` event is
not a successful end-to-end result.
