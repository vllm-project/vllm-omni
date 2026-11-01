# Hy-SimulMT streaming session contract

**Status: proposed; contract scaffolding only.** This is the first serving
increment for [RFC #9511](https://github.com/vllm-project/vllm-omni/issues/9511).
It adds types, policy/backend interfaces and a synchronous metadata state
machine in `vllm_omni/entrypoints/hy_simulmt.py`. It does not enable a model,
route or scheduler. Realtime wiring, queue execution, metrics and concurrency
benchmarks follow separately. Existing endpoints retain their behavior.

## Hypotheses and identity

A connection owns increasing `segment_id` values starting at zero. Within each
segment, admitted requests get increasing `revision_id` values starting at one;
IDs are never reused within a segment. `epoch` identifies translation context
and increases on reset. Engine `request_id` values must be globally unique,
including across reconnects (for example connection UUID plus revision tuple).

Every hypothesis carries these IDs, `status: partial | final`, and **complete
replacement** `text`. Empty text is a valid replacement. Clients replace the
display for that segment; they must not append the text or assume prefix
stability. A final may rewrite every partial and is authoritative. A boolean
`revision` flag alone cannot order delayed results and is unnecessary.

Partial publication is ordered by the latest **published** revision, not the
latest admitted request. An admitted partial may publish once if its revision
exceeds the publication watermark for the open segment. Queuing/coalescing a
newer partial does not invalidate a running result: otherwise inference slower
than the cadence could suppress every partial indefinitely. Publishing a newer
partial suppresses older completions. Final admission still invalidates all
partials immediately. Completed segments, old epochs, cancelled sessions and
duplicate callbacks cannot publish again. The owner only passes completions
from requests it actually dispatched; coalesced queued work has no callback.

## Segment and session lifecycle

| Operation | Contract |
| --- | --- |
| Append | PCM16 little-endian, mono, 16 kHz; explicit source language `zh` or `en`. Accumulate the complete current segment. |
| Partial | When enough new audio has arrived, request re-translation of the entire current segment. |
| Commit | Admit one final re-translation of all accepted segment audio; freeze that audio until the final finishes. |
| Final completed | Emit the final, release segment audio and start the next segment. Subsequent audio belongs to that new segment. |
| Input completed | Admit the current final, drain it, then close input; an empty segment closes without inference. |
| Cancel / disconnect | Invalidate results immediately, discard buffers/context, abort and await all engine work. Never synthesize a final or a successful completion. |

The scaffold uses `OPEN -> FINALIZING -> OPEN` for ordinary commit and
`OPEN -> DRAINING -> CLOSED` for input completion. Cancellation moves any
nonterminal state to `CANCELLED` and is idempotent. Append/commit/reset are
rejected outside `OPEN`. Empty ordinary commit is an error. A backend failure
is reported as an error and terminates the session through cancellation;
failed finals must never become successful completion events.
Use `is_current(revision)` to classify a backend failure, then cancel if it is
current. Ignore stale failures; do not use `complete()` as a validity probe.

**Finalization is a barrier in this first contract.** Audio arriving while a
final runs is backpressured, not assigned to the old segment or buffered in an
unbounded next-segment queue. A client retries an unaccepted append/commit;
the adapter must explicitly acknowledge acceptance. Completion while already
finalizing is likewise rejected/retried; it does not implicitly cancel work.

## Cadence, context and reset boundaries

`StreamingConfig` accepts integer milliseconds: default `cadence_ms=1600` and
`reset_ms=30000`, with `1 <= cadence_ms <= reset_ms <= 30000`. These are proposed
policy defaults, not checkpoint constants or a latency guarantee. The 30 s
upper limit respects the audio clip limit in the RFC. Integers exclude booleans
and nonfinite/fractional durations. Small cadences are valid but may coalesce
heavily. Cadence is measured in newly accepted audio samples, not wall time;
silence in the transport does not trigger repeated inference. One large append
makes at most one latest partial due, not a burst of historical partials.

Context duration counts accepted audio across committed segments. At exactly
the reset boundary, finalization takes precedence over a partial. Admit and
finish the old segment's final **before** incrementing the epoch and clearing
history. Even when cadence does not divide reset duration, no audio is lost.
An append crossing the boundary is rejected atomically: the adapter reports
the remaining sample budget so the client can split and retry it. No prefix
of a rejected append may be accepted. The final barrier applies before the
remainder can start a fresh segment/epoch. Explicit context reset is allowed
only at an empty, open boundary; otherwise finalize first.

The future adapter retains authoritative finals and the latest published
partial in the current epoch as prompt context, with a hard **16,384 Unicode
character** budget (trim oldest text first). Request context is an immutable
snapshot; replacing a hypothesis also replaces its context entry. A reset
clears all context and awaits abort/release of obsolete work before allowing
new engine dispatch. Already emitted finals remain visible to the client.
Epoch/segment/revision checks reject late callbacks regardless of whether
engine cancellation wins the race. The scaffold accounts for samples and
identities only; buffer ownership and cleanup belong to the adapter.

## Admission and inference interfaces

`PendingWorkPolicy.admit()` atomically reserves capacity. Its outcomes are
`ACCEPTED`, `COALESCED` (replace a queued partial for this segment), or
`BACKPRESSURE` (no state change). `release()` returns a reservation only after
completion or acknowledged cancellation. The owner validates the operation
and cadence before reserving, then allocates the revision and enqueues its
snapshot synchronously, with no intervening await. Empty input completion
needs no reservation. If snapshot construction fails, release the reservation
and terminate with an error, rather than leaving an admitted final stranded.

`max_pending_requests=2` by default, configurable from **1 through 8**. It
counts queued plus running requests, including aborts awaiting cleanup. At
most one request runs and at most one latest partial waits per session. The
cap is a ceiling, not a target concurrency. Replace a queued partial with the
latest snapshot instead of building a backlog; release the replaced snapshot.
A running partial remains publishable until a newer result publishes or a
final is admitted; do not abort it merely because a newer partial was queued.
Obsolete running work still counts until released.
A final can replace a queued partial, but its outcome must be explicitly
`ACCEPTED`; it is never coalesced away. If no slot is available, return
`BACKPRESSURE` and leave audio, phase and revision counter untouched. Retry
after capacity is released. No successful commit acknowledgement precedes
admission. With cap one, a running partial must finish or acknowledge abort
before a final is admitted.

Audio storage is bounded separately: at most `reset_ms * 16` samples in the
live segment and per immutable pending snapshot (two bytes/sample). This is
at most 960,000 bytes per buffer at the default reset. The adapter must also
bound transport messages before decoding and retain no rejected audio.

`InferenceBackend.infer(InferenceRequest)` asynchronously returns one complete
string or raises. Requests carry immutable PCM, language, context, engine ID
and revision identity. `cancel(request_id)` is asynchronous and idempotent,
including for completed/unknown IDs; it must abort engine generation and await
resource release. Cancelling a Python task alone is insufficient. The owner
releases queue slots exactly once, suppresses stale results and clears all
audio/context on disconnect. Scheduling, task tracking, timeouts and the real
backend implementation are intentionally deferred.

## Realtime compatibility (proposed adapter)

The installed upstream speech-to-text protocol has
`input_audio_buffer.commit` with `final: bool = false`, append-only
`transcription.delta`, and server-to-client `transcription.done` with `text`
and optional `usage`. None can represent revisable segment text faithfully.
Ming3's manual-turn connection is useful transport precedent but has different
audio rates and response semantics.

- Reuse `input_audio_buffer.append` and `input_audio_buffer.commit`.
  `final=false` seals a segment; `final=true` means graceful input completion.
  `transcription.done` remains server output, never a cancellation command.
- Negotiate a model-specific session extension, proposed
  `session.translation={language, cadence_ms, reset_ms, max_pending_requests}`,
  with `audio.input.turn_detection=null`, PCM16 at 16 kHz and text-only output.
  Freeze these settings after audio starts. Do not change shared upstream
  schemas or silently activate replacement semantics for ordinary STT clients.
- Add an explicitly negotiated `transcription.segment` event carrying
  `epoch`, `segment_id`, `revision_id`, `status`, `text`, and
  `text_mode: "replacement"`. This is an extension, **not** an already supported
  OpenAI event. Do not emit replacement text as `transcription.delta`.
- Use `input_audio_buffer.committed` only after final admission, identifying
  the segment and echoing `request_event_id`. The final segment event
  acknowledges that input can resume. Empty graceful input completion emits
  `transcription.done` with `request_event_id` instead of a commit acknowledgement.
  Append acknowledgements and rejection fields are specified below.
- Map explicit `response.cancel` to terminal session cancellation in this
  text-only mode. A new session is required to resume. On graceful completion,
  emit the last final segment then `transcription.done` with `text` equal to
  that last segment's final text (empty if the session had no audio). Document
  this summary as non-additive; clients build the full translation from final
  segment events. No unbounded session transcript is retained.

These wire choices need review before the adapter lands. This PR does not
serialize events or claim endpoint compatibility tests have passed.

### Audio command acknowledgement and retry

In the negotiated translation mode, every append/commit carries a client
`event_id`: a nonempty string of at most 64 characters, unique per command
attempt within the connection. Validate it before consuming input. Server
responses echo it as `request_event_id`; this is correlation, not a server
deduplication/replay cache. Retries use a new ID. These fields and the appended
event below are negotiated extensions; shared upstream STT schemas stay intact.

After atomically accepting an entire append, send exactly one
`input_audio_buffer.appended` acknowledgement before any hypothesis triggered
by that append. Its fields are `request_event_id`, `epoch`, `segment_id`,
`accepted_samples` (for this command, not cumulative), and `remaining_samples`
(reset budget after acceptance). For example:

```json
{"type":"input_audio_buffer.appended","request_event_id":"a1","epoch":0,"segment_id":0,"accepted_samples":25600,"remaining_samples":454400}
```

A rejected command gets exactly one `error` and no success acknowledgement.
Preserve the upstream string `error` and `code` fields; add `request_event_id`,
`retryable`, `epoch`, `segment_id`, and `remaining_samples` (unchanged budget).
Use `code="backpressure"` for admission/finalization barriers and
`code="audio_boundary"` for crossing appends, both with `retryable=true`.
Invalid IDs/audio/configuration are non-retryable validation errors; if the
ID itself is invalid, `request_event_id` is null. No rejected audio is consumed.

```json
{"type":"error","error":"Append crosses reset boundary","code":"audio_boundary","request_event_id":"a2","retryable":true,"epoch":0,"segment_id":0,"remaining_samples":19200}
```

Clients use stop-and-wait for audio commands: wait for an append acknowledgement
or rejection before sending the next append or commit. On boundary rejection,
split the rejected PCM at `remaining_samples`, send the fitting prefix with a
new ID, then commit and wait for the final before sending the remainder. If
the budget is zero, commit first. On backpressure, retain the command's audio
and retry with backoff (or after the pending final event). Never retry an
acknowledged append. Pipelined audio commands are outside this first contract;
correlation alone cannot preserve audio order when a later append is accepted
before an earlier rejected append is retried. If acknowledgement is lost due
to disconnect, start a new session; cross-connection replay is not supported.

## Validation and subsequent work

CPU unit tests pin duration validation, sample boundaries, admission rollback,
revision ordering, sustained slow-inference progress, final barriers, empty
input, reset and cancellation. A
controllable fake backend exercises late completion and cancellation without
sleeps or model weights. Subsequent scheduler tests must cover capacity
release, partial coalescing, backend errors and disconnect cleanup; transport
tests must cover negotiated schemas and acknowledgements. Model parity and
concurrency 1/4/8 measurements belong to the later integration: report queue
delay, inference and final/partial latency separately from translation quality
and StreamLAAL. GPU/Kaggle testing is not needed for this metadata scaffold.
