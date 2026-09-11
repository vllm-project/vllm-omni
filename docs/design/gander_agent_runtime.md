# Gander context and tool runtime

The Gander integration builds on PR #7294 at
`36142ba106cdf458dfc77439cc8f769a68d1892d`. That PR supplies transactional
Stage0 append, bounded input journals, physical request generations, deadline
handling, and MiniCPM replay. This integration adds model-owned history
selection and an explicit context replacement transaction. It does not execute
business tools or connect a Brain provider.

## Ownership

| Layer | Responsibility |
| --- | --- |
| Gander adapter | Tool grammar/schema validation, silent tool units, `[SLATE]`, protected system/tools/reference prefix, history unit selection, retained assistant output tokens, replay framing and exact token budgets. |
| Duplex serving | Ordered control input, client validation, playback cancellation, epoch transition, receipts, and rejecting old generation output. |
| Engine control plane | Preflight model plan and memory/context bounds; retire old physical requests; replay on a new request; publish success only after reconstruction completes. |
| Scheduler/worker | Own and release KV blocks; rebuild contiguous positions through normal prefill; retain real multimodal embeddings for preemption recomputation. |
| External application | Execute tools, bind trusted user turns, maintain persistent task revision/cancellation semantics, and deliver results. |

The engine contains no Gander token IDs, slate syntax, or hardcoded 128-unit
window. `DuplexContextUnit`, `DuplexContextOutput`, and `DuplexContextPlan` are
model-neutral records. Optional adapter hooks produce and consume those records.

## History and protected prefix

Each committed input append has a stable `unit_id`. A unit contains its input
and, once completed, the exact assistant action/text/tool token sequence.
Fast model completion before append-receipt publication is buffered until the
journal entry exists. Incomplete generated output is not replayed. On explicit
replacement, output belonging to the active interrupted response is discarded;
previously replayed completed history remains intact across repeated replacement.

Gander's default history policy triggers rollover at 128 units and retains 96.
Tests can set a smaller window through `extra_body.gander_history`.
System instructions, tool schemas, reference voice, and the latest slate are
always reconstructed as the prefix. They are not removable history units.
Up to 16 history units may additionally be pinned, subject to the configured
window. Explicit deletion of a pinned unit requires unpinning it first.

Automatic rollover waits until the active response has been delivered, so
its trailing audio and terminal event cannot be discarded with old KV. The
unit threshold may temporarily be exceeded during a reply; hard journal
token/byte and scheduler context limits still apply and fail explicitly.
Explicit context replacement continues to cancel the active response.

For rollover, the policy retains pinned units and a recent suffix, preserving
relative order. The generic engine can remove additional unprotected units to
fit token/byte limits, but fails if only protected content remains too large.
If capacity pressure arrives before the unit threshold, selection also frees
at least a quarter of the unprotected window rather than retaining nearly
everything and rebuilding on each append. The deployment allows a bounded
256 MiB replay journal to account for repeated reference-prefix snapshots.
The first retained unit is re-budgeted with the current prefix; assistant
history is teacher-forced, not sampled again. Replay cannot call tools or emit
historical speech.

This reconstructs KV; it does not splice or move raw KV tensors. Audio/vision
features are re-encoded from retained inputs in the selected order. Reordering
or dropping audio changes frontend context, so bitwise equivalence to the
original run or the upstream custom decoder is not claimed. Pinning separated
units may also produce discontinuous audio history; applications must choose
history edits deliberately.

## Replacement transaction

1. The adapter validates event identity, versions, schemas, and inserted data.
2. Serving stops scheduling automatic continuations and waits for preceding append
   receipts without cancelling committed operations. `context.validate` checks the proposed history order, prefix,
   and memory budget before old KV is retired.
3. Under the serving event lock, active playback is cancelled and the epoch
   advances. Old model output fails the epoch fence. The protocol emits an
   explicit `output_audio_buffer.cleared` for an active cancelled response.
4. `context.replace` checks that the validated input/configuration generation
   is still current, retires all old physical stage requests, and replays the
   selected units with a new physical request identity and contiguous positions.
5. The engine waits for replay to reach its safe completion boundary. Only
   then does `input.context.replaced` confirm the new epoch, context version,
   physical request, retained/deleted units, and token count.
6. Inference resumes from the new context. `generate:true` requests a model
   decision without fabricated microphone samples. Slow replacement uses a
   bounded deadline; it does not promise constant latency.

Validation failure leaves old KV intact. Failure after the epoch transition
fails closed and requires session reopening; there is no rollback to a
partially retired physical request. Identical replacement event IDs return the
committed receipt without another rebuild, including retries carrying the old
epoch. Conflicting IDs fail. Replacement intentionally retains external call
identity across the new frontend epoch; it does not cancel the background task.

Old transport journal entries are retired after replacement. A reconnect cursor
older than that boundary requires resynchronization instead of replaying stale
audio. Bytes already delivered to a speaker cannot be retracted by the server;
the client must honor the playback-clear event.

## Appends versus replacement

Tool results and background progress normally append escaped `<tool_response>`
blocks in input order. They can opt into replacement/preemption. A slate update
always replaces the protected prefix; old slate observations no longer remain
in current KV. History edits support insertion of validated external events,
move, delete, pin, and unpin. Absolute raw token/KV editing is not a client API.

`input.context.appended` means queued. `input.context.applied` is a cumulative
prefill watermark from model output. `input.context.replaced` is stronger: it
confirms the replacement transaction and new physical context are ready.

## Limits and external runtime

Six tool schemas / 1024 schema tokens; one silent tool call per unit / 256
generation tokens; 1500 tokens per external observation; 256 tokens per slate;
32 edits / 64 KiB per replacement; bounded call and event receipts. Engine
replay token/byte budgets remain authoritative even when a unit limit is larger.

Gateway, Brain, trusted ASR-to-task binding, persistent task storage, actual
business tools, and task-run revision generations remain outside this scope.
Model epoch fencing does not replace business-level stale-result checking.
Paper agent metrics require those external components and their full scoring
protocol. Current H200 evidence is recorded separately from implementation
claims; no arbitrary hardware/concurrency or performance certification is implied.
