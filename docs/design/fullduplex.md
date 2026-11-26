# Unified Full-Duplex Framework

This document is the design record of the full-duplex serving framework:
the engine-resident session model, the typed command/event contract, the
single model plugin, and the serving/API surfaces built on top of them. It is
the owner document for `vllm_omni/engine/duplex/**`,
`vllm_omni/engine/duplex_omni_engine.py`,
`vllm_omni/engine/duplex_orchestrator.py`,
`vllm_omni/entrypoints/duplex_omni.py`, `vllm_omni/entrypoints/duplex/**`,
`vllm_omni/clients/**` and the per-model `duplex/` packages.

`vllm_omni/protocol/realtime/**` (the OpenAI Realtime API) and
`vllm_omni/protocol/duplex/**` (vLLM-Omni's extension of it) are the shared
wire codec duplex is built on. Duplex is its first consumer but not its owner: the codec
is deliberately runtime-agnostic so a second Realtime surface can reuse it
(RFC #6592 P0a).

The public wire contract lives in
[`docs/serving/realtime_duplex_api.md`](../serving/realtime_duplex_api.md);
this page describes how it is produced.

## Purpose

A full-duplex model listens and speaks at the same time: audio streams into
the model while it produces audio, the model decides when to speak, and the
user can interrupt. Serving such a model needs a long-lived session with its
own identity (epoch, turn), ledgers (input, response, playback, history), a
lease (idle TTL, disconnect grace, resume) and a binding to the resumable
stage requests that carry the session's audio through the pipeline.

The framework gives that session exactly one owner, keeps the generic
turn-based stack free of duplex code, and exposes the same session to three
consumers: the OpenAI Realtime WebSocket route, the in-process Python API and
the model plugins.

## Runtime process placement

The optional process runtime isolates the orchestrator from the frontend GIL
and serves sessions only. See [process runtime usage, ownership, and limits](../serving/full_duplex_api.md#run-the-duplex-runtime-in-a-separate-process).

## Incremental text input

`AppendText` is admitted only when the model advertises `text` as an input
modality. MiniCPM-o consumes each fragment as a native TEXT unit in automatic
response mode. The manager rejects text that cannot be encoded as UTF-8 with a
request-scoped `bad_event`, then charges UTF-8 bytes before mailbox admission; the
runner keeps that charge until its ordered append task completes or is
cancelled. Text shares the audio/video append tail and epoch fencing, without
flushing partial PCM or advancing the first audio-encoder window. Before queuing
text, the runner submits any complete buffered media units through that same
append tail, preserving their input flags and leaving incomplete PCM buffered.
Eligible audio appends also submit every complete buffered unit on either
clock, including appends arriving after text has opened the response. An
overlap policy that defers an append still keeps its media buffered.

The MiniCPM plugin tokenizes off the owner loop and reserves exactly the
context/previous-unit closure, `<unit>`, and text tokens. Stage 0 embeds these
tokens using the native `streaming_prefill(text_list=...)` layout and retains
them in sliding-window units. Repeated prefill of the same append identity
reuses its embeddings. `force_listen` permits intermediate fragments to enter
the KV cache without generating a reply; an unforced text unit opens the
assistant turn once during prefill, since text has no microphone silence
marking the end of input. When no response is active, the runner creates one before submission;
an empty `<|chunk_eos|>` unit returns directly to the session to advance the
normal continuation clock instead of waiting for nonexistent Talker output.
This shortcut applies only to model-confirmed unforced text input. Empty audio
units retain their Talker handoff and segment boundary, including when the
chunk terminator is their only token, so input-clock acknowledgements keep
their unit ownership.
For a response released by unforced text, incomplete buffered PCM does not
block that clock. The PCM remains charged and unconsumed; queued appends,
reserved input, and complete audio or vision units retain priority over
generated silence. Stage 0 confirms this exception for the epoch and model turn
that consumed the text, using the bridge's turn snapshot before handoff. The
active response at text admission may still belong to the previous Talker turn;
it must not receive the new turn's continuation permission. An empty first unit
waits for the previous response's final audio and `response.done` before opening
the new response and continuing. Cancellation discards this deferred unit. The exception does not
carry over to a later audio-only response or a new epoch. Disabling silence continuation or
using the input clock still prevents synthetic units.
An empty Talker segment, such as a unit containing only `<|speak|>` and a
chunk terminator, uses the same deferred response transition when Stage 0 has
confirmed its text turn. It retains its normal handoff and segment boundary;
the session binds continuation to that text turn even if the previous response
has ended before the empty segment arrives.
For units returned directly to the session, the Talker handoff skips their
completed control tokens in cumulative output before aligning the next speech
unit's text and hidden states.
Replayed prefill cannot reopen an ended reply. This does not change the separate
conversation-item text-turn capability or replace assistant playback controls.

## Playback completion ownership

Cancellation of a retained model stream identifies the response owning its
queued audio, even after generation completes. The Realtime projection clears
that playback without emitting a second terminal for the completed response.

Deployment output capabilities are resolved after `configure_pipeline` using
`deployment_capabilities`. A topology consisting of one final text stage
advertises text output only and no playback ACK, audio truncation, or stage
handoff. Other topologies retain their model plugin's output contract.
`DuplexCapabilities.emits_audio` is the shared deployment predicate for output
projection, history, synthesis parameters and calibration. Both typed and
native truncation commands use one control-layer check of
`supports_audio_truncate`, including audio-capable models that disable it.
The engine capability property and session admission share this resolution.
Startup warmup uses the engine's output defaults and, for text output, seeds
an opening user turn and waits for a completed response containing text.

`DuplexSessionConfig.modalities=None` preserves omission until admission.
Both wire aliases are validated before defaults; unavailable output is rejected
before reserving a session replica or creating a stage request. Live updates
and response overrides validate a candidate before adopting configuration or
reserving response options. Modality names are normalized to lowercase before
validation. Explicit empty lists, normalized duplicates and conflicting aliases
fail.

A text-only deployment projects model results through existing text events.
Its data plane still owns native unit and turn boundaries. Text responses
reserve their history position before continued input, commit emitted text
on completion or cancellation regardless of playback policy, and release
playback bookkeeping without ACKs. Whitespace-only output creates no assistant
history item or snapshot; meaningful emitted text is preserved verbatim.
The unconditional bypass is deployment-scoped. On audio-capable deployments,
a reply with zero generated/sent audio and no audio-completion dependency
commits text without an ACK through the zero-audio fallback. Replies whose
text depends on complete audio still follow alignment and playback checks.
Selecting text output alone does not implement per-request TTS skipping.
Cancellation rejects old-epoch results
through the existing fence. Audio calibration is not started for deployments
without audio output.

MiniCPM-o 4.5 registers `minicpmo_4_5_thinker_only`, a Stage 0-only pipeline
selected by a deploy config's `pipeline` key. Its data plane sends Stage 0
output as text and ends a unit at LISTEN, `chunk_eos` or `chunk_tts_eos`; a
unit ends the turn only when its final token before the terminator is
`<|turn_eos|>`. Later text or a speak decision keeps the response open.
The session requires a fast tokenizer and decodes cumulative tokens
in unit order, so a snapshot spanning turns keeps each turn's text separate.
Stage 0 metrics are cumulative snapshots under
`metadata.vllm_omni.stage_metrics`. `response.speak` carries the running
totals and text deltas omit them; the event that ends the response carries the
final values: `response.done`, or `audio.cancelled` when it is cancelled.
A snapshot that spans two turns is charged to the first reply, since the
engine reports one set of metrics per snapshot.
The first text delta carries server TTFT under
`metadata.vllm_omni.response_request_metrics`, including responses already
opened by `response.create` or `input_audio_buffer.commit`.

`DuplexEngineSession` owns response snapshots and playback cursors. Output
completion and a client's full `playback.ack` can arrive in either order.
On all deployments, continuation-limit endings discard uncommitted text and
release response state if the reply produced no audio. Failed responses and
`conversation: "none"` endings also release zero-audio state without committing
text to history. On audio-capable
deployments, a later ACK for that released response fails with
`playback_item_not_found` without restoring its state. Text-only deployments
reject ACKs with `unsupported_operation`. The built-in playback helper skips
finished replies with no audio.
`finalize_response_playback` is the shared check used by ACK processing,
normal response completion, draining response completion, and cancellation.
It retains state while that response is active, any of its requests are still
draining, or playback is partial without an explicit hard truncate. A hard
truncate means playback has stopped; once output ends, no further ACK is needed.
At that point, detach the stopped response from the current playback gate even
if calibration keeps its response-scoped cursor. Detach by cursor identity so
an older stop or delayed ACK cannot unblock or overwrite a newer response.
A final output
can add transcript or completion metadata after the ACK, so history is reconciled
against the retained cursor before releasing the snapshot and pending item state.
Deleted history is never reconstructed. Natural drain completion establishes
`audio_complete`; cancellation does not. A later response's cursor and history
position remain independent. Without calibration, aligned history keeps its
timing marks after a hard stop so a later explicit truncate can shorten the
stored prefix. Marks retain their original character offsets; repeated truncation
does not rescale them to the remaining text. Unaligned stops and retired
calibration evidence keep only the stored prefix and cutoff. Delayed progress
ACKs cannot change finalized history or restore unheard text. Plugin runtime
settlement follows all history writes, including cancellation writeback into a
reserved older turn; it must observe the final committed result.

Audio truncation is valid only for an item whose response produced audio.
The session checks its live cursor or retained audio provenance before any
history or playback mutation. Cursor cleanup preserves that provenance only
for stored history items; deleting or replacing finalized context removes it.
This distinguishes a retired audio reply from imported context or a reply that never emitted audio
without retaining its recording or full response snapshot. Realtime commands,
native truncate signals, and hard playback ACKs reject a no-audio target.
An ACK's implicit target is resolved before validation or mutation. Pending and
stored-item fallbacks must match `last_response_id`, which `begin_response`
updates together with the current playback cursor. A retained older reply cannot
consume the current reply's ACK, even when it is the only remaining history item.
Resolution looks up that current item directly in pending or stored history;
unrelated older or imported items do not make its identity ambiguous. This must
continue working when a partial ACK moves it from pending to stored history.
The stored-item fallback also requires the item's live or retained audio
provenance, including for audio-only replies, then passes the same ownership
and late-ACK validation as an explicit target. Imported context without audio
provenance is not selected, even if its ID starts with `item_`. Without an
identifiable target, ordinary ACKs
update only the current playback cursor and hard ACKs fail before any history or
cursor mutation.
The typed `AckPlayback` command preserves the wire `truncate` flag through
translation and mailbox delivery, so hard stops reach the same handler as
internal playback controls.

History edits pass the item's own timing marks and playback cursor explicitly;
the text-prefix calculation never reads the active response implicitly. If both
marks and duration of a known audio reply have been released, a new explicit
truncate cannot establish a heard prefix and clears the text conservatively.
Truncation preserves the empty assistant turn so the command's following playback ACK remains valid;
only deletion removes that turn. Ignored ACKs for finalized history report
`history_committed: false` and do not settle plugin runtime again, including
when calibration evidence has been retired.

Opt-in history calibration is a separate retention owner: incomplete or
hard-capped replies remain in its bounded cache for prefix correction. Every
completion path preserves that evidence until calibration retires it. Sessions
without calibration release stopped output immediately.

## Experimental Qwen history policy

Qwen defaults to `full_history` when `history_policy` is omitted from server-owned
`duplex_session.model_custom_options`. It sends all effective Session messages and
retained original input audio/images, without a recent-message window, automatic
summaries, media substitution or oldest-first eviction. Source deletion and
playback calibration still determine the effective Session records.

| `history_policy` | Model input | Capacity behavior |
| --- | --- | --- |
| `full_history` (default) | All effective original messages and media. | Reject the request if the complete input plus reserved output exceeds resolved token or per-request media limits. |
| `recent` | The legacy 16-message window and bounded input-audio cache. | Preserve the previous clipping behavior. |
| `bounded_recent` | Recent complete turns under the configured context/media budget. | Remove older turns until the request fits. |
| `incremental` | Published memory plus selected uncompressed turns. | Compress older sources and fit the foreground request to budget. |

The default now changes from `recent` to `full_history`. Deployments that need the
previous bounded window must explicitly set `history_policy: recent`. Full-history
requests preserve system-message order and the existing tool-call protocol. They
keep pending tool-call anchors beyond the legacy 16-message window. Full history
uses original sources even when `context_keep_media_turns` is configured; select
`recent` to use that policy's media-dropped view.

The plugin checks the deployment's per-request audio/image counts and passes its
context limit and reserved output to the stage submission. The engine renders
the input once, checks those actual expanded tokens before backend admission,
then passes the same rendered input and media features to `InputProcessor`.
Full-history validation does not run a separate HF processor pass. The context
limit includes `max_context_tokens` when configured. Overflow
raises a recoverable `input_capacity_exceeded` error without submitting a partial
request or deleting history. Clients can delete source items and retry. Session
input-admission limits still reject additional material before it is accepted.
The base deployment permits four audio inputs and eight images per request;
full history does not raise these deployment limits automatically. Direct plugin
calls without a resolved model configuration leave validation to the downstream
engine.

Session-owned input readiness records authoritative history source IDs, including
rejected requests; unbound seed text keeps its existing notification count.
Deleting/recreating an item cannot rearm its old identity, while transcript
corrections retain that identity. Text and speech arriving during preparation
remain independently pending. The model deletion hook removes idle committed
PCM before a reused item ID can merge with the deleted source, and the Session
releases the corresponding reserved input bytes. A queued append retains the
same payload identity until consumption settles.

Full history retains original input PCM without the recent cache's item/byte
evictions. Merged commits preserve their item boundaries: deleting one source
removes only its PCM segment during the next prompt preparation. New commits
remain separate from previously rejected input. Closing the Session releases its
state. Storage can grow with the conversation, and the original history is counted
again for each request.

Every built-in, including `full_history` and `recent`, is a `QwenHistoryPolicy`
object. Their `select` methods choose the original groups to render.
`HistoryBehavior` declares original versus summary-aware rendering, the overflow
action, message and media retention limits, and whether an L1 media view applies.
The plugin consumes these choices without comparing the configured policy name.
`FullHistoryPolicy` selects all original groups and rejects overflow;
`RecentHistoryPolicy` selects the legacy window and declares its cache limits.
Custom original-history strategies can subclass either and override selection
or behavior. Session image admission is supplied by `session_image_limits`.
Existing custom bounded/compression strategies retain the four required hooks.

Audio retention is selected explicitly with `audio_retention`. `original` keeps
PCM until source deletion and requires both audio cache limits to be `None`.
`bounded` requires a positive `audio_cache_items` limit and accepts an optional
`audio_cache_bytes` target; it keeps at least the newest binding, including when
that binding exceeds the byte target. Unsupported combinations fail at policy
construction. With original-message rendering, `original` also rejects selected
sources whose PCM is unavailable, independently of the context overflow action.
This check uses the selected source before L1 view formatting; a transcript view
cannot hide missing PCM and does not release retained original audio.

`DuplexAppendPlan.input_capacity` carries model-owned context and output limits
through `DuplexStageSubmission`. `DuplexOrchestrator` owns renderer adaptation and
checks expanded input before ordinary backend length validation and submission.
The shared worker delegates input processing and retains its serialization,
cancellation join and sender-cache cleanup. Capacity checks count prompt
embeddings as well as token IDs. Other plugins leave this optional field unset.

`incremental` folds older text and original audio into a bounded text memory,
and images into intermediate and long-term visual memories.
Model-specific selection, source retention, compression prompts and token budgets
remain in the Qwen plugin. Published dialogue and visual summaries are `SUMMARY`
nodes in `DuplexEngineSession.context_ledger`; the public history snapshot supplies
their source IDs and revisions. Qwen and JoyAI share `engine/duplex/text_folding.py`
for labeled text excerpts and `engine/duplex/visual_memory.py` for visual batching
and retention. `SessionVisualHistory` in `engine/duplex/session/visual_history.py`
binds visual sources to the public snapshot, resolves valid ledger text, and
publishes/retires visual summaries. Models supply decoded batches and prompt
formatting. `SessionMemoryRuntime` executes private requests with the Thinker
as both entry and final stage.

Each Session holds `QwenDuplexSessionState.history_policy_state`. This state keeps
summary node IDs, selection progress, retained media and unfinished text excerpts.
Summary text is read from the Session ledger. Qwen's visual store keeps batch
metadata; its published text is also read from ledger nodes. Original Session
records and source dependencies remain retained, so the bounded prompt and media
limits do not bound total Session storage.

An assistant reply reserves a history position before becoming visible. When
generation or playback materializes that reply, unchanged camera sources can
move to later list positions. Their published visual summaries remain valid
while source IDs, revisions, image parts and URLs are unchanged. Pending
compression work still validates its original message labels and is retried
when those positions move.

Unanswered user items and intervening system messages form one protected turn.
Merged audio submissions retain their committed source IDs. Only the source
items whose PCM is present share that payload in the prompt, including on the
explicit `recent` path. Deleting an assistant reply can join older turns without
merging their existing audio bindings. Distinct bindings remain separate even
when their payload objects or PCM values are identical.
Foreground fitting preserves that complete turn, including original media, or
returns a recoverable capacity error. After removing oversized source items,
clients can retry the remaining input without submitting a new user item.
Background dialogue and visual compression wait until the turn has an assistant
reply. `chunk_messages` is a batch target:
an indivisible audio turn can exceed it when its actual audio count, retained
bytes and processor-expanded input plus reserved output fit the hard limits.

The summary lifecycle is:

1. The Session owner supplies a current snapshot and refreshed ledger through
   `bind_history_context`. Qwen selects sources by position in that snapshot and
   captures their public `(source_id, revision)` references.
2. Qwen's compression policy prepares one private Thinker request. The existing
   `SessionMemoryRuntime` handles submission, cancellation, cleanup and retry.
3. Before applying its result, the runtime supplies the current snapshot again.
   Qwen checks source references and its selection progress, validates the model
   output, and publishes through `ledger.replace_summary` on the owner loop.
4. A rolling dialogue summary inherits all dependencies of the previous summary
   plus the new batch's sources. A long-term visual summary inherits all sources
   of its intermediate summaries. Replaced nodes are retired after publication.
5. Before foreground preparation, the owner refreshes the ledger against current
   history. Qwen reads surviving nodes and applies its own selection and token
   budget. Deletion, correction, transcript or playback revision changes revoke
   affected summaries, including when a tighter cutoff leaves visible text equal.

Replacing summaries flattens their source dependencies, so obsolete summary
nodes do not accumulate into a parent chain. Qwen rewinds its dialogue selection
progress after invalidation; an invalidated visual batch resets its visual plan.
Evicting a long-term visual summary also removes its ledger node. Visual writeback
first stages batch/cursor changes using copies of the small metadata lists, sharing
immutable encoded payloads. Only successful ledger publication adopts that state;
either validation failure leaves the previous visual store and ledger intact.
L1 nodes remain independent. A missing audio source records an omission and does
not block later work; the omitted source is not claimed as summary evidence. Media eligibility
remains independent of the recent text window.

External policies inherit `QwenHistoryPolicy` and supply four methods:

| Method | Decision |
| --- | --- |
| `select(state, limits)` | Original whole turns offered to the next foreground request; the runtime adds stored memory and fits the actual input. |
| `next_batch(state, limits, *, max_audio)` | Next unprocessed prefix to compress, or `None`. |
| `build_compression(batch, previous_memory, limits)` | Compression instructions, quoted source layout and temperature for a private Thinker request. |
| `parse_compression(text)` | Complete replacement memory, or rejection of the model output. |

For example, an extension can reuse incremental selection/batching while
replacing free-form summaries with a validated JSON fact list. Compression
instructions and result interpretation belong to the policy. The runtime owns
PCM binding, Qwen processing, actual token limits and committing valid results;
`SessionMemoryRuntime` still owns submission, cancellation and retry. The
extension does not implement a second model-execution loop.

Compression builders run in the joined CPU preparation worker on frozen
`CompressionMessage` values. Text, batch-local `CompressionAudioRef` and
`CompressionImageRef` indices replace borrowed Session objects and media. Every
reference must appear once in source order within its modality in the returned
`CompressionPrompt`. The same compression hooks construct and parse visual
summaries. `compress_visual_history=False` disables visual summarization;
`bounded_recent` uses this setting. If a batch must shrink
to fit context, the builder runs again on that actual smaller batch. Parsers
run on the Session loop and must be small and nonblocking. Both hooks are
stateless, perform no I/O and must not retain input material. Selection hooks
receive Session-owned state under a read-only contract; server-loaded Python
extensions are trusted code, not a sandbox.

Before invoking the parser, the runtime rejects stale, empty, truncated or
overlong model output. It also bounds the parsed text before committing memory
and source progress together. A parser can return `None` or raise `ValueError`
to reject malformed output without advancing progress. Memory remains bounded
text (including serialized JSON), inserted as reference data in foreground
input. The interface uses the same Qwen Thinker; it does not support external
summarization services, arbitrary retrieval stores or mid-session switching.

The built-in incremental parser also rejects output containing only source
delimiter markers, role labels, numbered visual-source labels and the closing
instruction `Return the complete updated memory.` Rejection preserves published
memory, pending images and text, and compression progress for retry. Labels
accompanied by actual facts remain valid memory. These checks detect structural
echoes; they do not certify factual accuracy. Control-line matching ignores
surrounding whitespace but otherwise requires exact case and punctuation.

The default compression prompt separates role labels from source text. For
images, it requests observable content even without a stated user preference
and combines repeated observations across frames into one memory.

Configure the policy in a deployment overlay:

```yaml
base_config: /path/to/vllm-omni/vllm_omni/deploy/qwen3_omni_duplex.yaml
duplex_session:
  model_custom_options:
    history_policy: incremental
    history_limits:
      recent_messages: 8
      chunk_messages: 8
      summary_tokens: 512
```

Pass the overlay with `vllm serve ... --omni --deploy-config /path/to/history.yaml`.
Set `history_policy` to `bounded_recent` for clipping, or to
`your_package.policy:YourPolicy` for a server-importable class with a no-argument
constructor. Subclass `IncrementalHistoryPolicy` to reuse its selection and
batching while overriding compression. Omitting the policy selects `full_history`.
The optional `history_limits.max_context_tokens` caps context below the resolved
model limit, including reserved output; omitting it uses the model limit.

Message count alone cannot distinguish a short text turn from a large image or
audio turn. Optional token targets let deployments trigger compression according
to the cost of the originals that remain in context, while keeping the same
selection/compression interface. They independently replace message-count
triggering and batching:

```yaml
base_config: /path/to/vllm-omni/vllm_omni/deploy/qwen3_omni_duplex.yaml
duplex_session:
  model_custom_options:
    history_policy: incremental
    history_limits:
      recent_tokens: 6000
      chunk_tokens: 2000
      summary_tokens: 512
```

`recent_tokens` starts dialogue compression when unsummarized originals exceed
the target. Once triggered, the policy fills a batch up to `chunk_tokens` from
the oldest completed turns. For example, roughly 7000 original tokens can
produce a batch of roughly 2000 tokens, leaving roughly 5000 original tokens
alongside the new summary. The output size is controlled by `summary_tokens`.
Reaching the trigger does not require a full batch to be available.

Both token targets default to unset. Setting `recent_tokens` replaces
`recent_messages`; setting `chunk_tokens` replaces `chunk_messages` for batch
size. Either can be used alone with the other message-count setting. Targets
must be positive integers. Audio count/byte pressure and visual count pressure
remain independent triggers, including below `recent_tokens`.

Counting runs in joined CPU workers. Dialogue targets sum the rendered token
costs of complete original turns, including their chat templates and expanded
audio/image tokens. Existing summaries, compression instructions and reserved
output are excluded from these source targets; the final rendered request
must still fit all of them within the effective model-context limit. The shared `SourceTokenCache` stores only scalar costs, public source references
and model-supplied media-view keys. Unchanged turns reuse their costs after an
append; edits, regrouping, audio rebinding and visual summary/omission changes
require a fresh count. Processed and removed groups are pruned. Token counting
adds CPU preparation work and does not bound total Session storage.
Visual batches also shrink to `chunk_tokens`, within their existing image-count
and summary-group targets. Indivisible audio turns and visual entries may exceed
the soft batch target when they fit the hard model/media limits. An oversized
text turn uses the existing private excerpt-folding path, even when the whole
turn would fit the model context. No partial excerpt publishes summary coverage.
An unusably small `chunk_tokens` value can leave no room for excerpt labels;
preparation logs the capacity issue and keeps the original source pending.

Incomplete implementations fail at server initialization. Selection is
stateless and may be cached while
the Session history revision, memory generation, retained-media revision and
audio-preparation ownership are unchanged. Reusing an already retained audio
payload for counting does not change its retained-media revision. Source checks still run before submitting or applying a summary.
Audio compression uses the renderer's effective per-request audio count and a
separate `audio_summary_trigger_bytes` setting (default 8 MiB). The trigger is
capped by the strategy's `max_audio_bytes` retention limit (default 32 MiB),
independently of the legacy recent cache. The policy receives the resolved
per-request count as `next_batch(..., max_audio=...)`. A still-valid foreground
source can be recovered for compression, but its worker references must fit the
strategy limits and remain charged until
feature preparation finishes. Current text/audio/image input is preserved independently of
background retention capacity; unavailable current audio is an explicit error.

Foreground rendering emits one debug record per successful prompt build with
`degraded`, `trimmed_turns` and `summary_dropped`. These count preparation-time
clipping, including plans cancelled before submission, rather than completed
responses. Worker logging does not mutate Session state.

When an indivisible text turn exceeds the compression input capacity, the
runtime folds labeled excerpts using private `HistoryFold` state. Each request
includes the previous intermediate memory. Only the final successful excerpt
publishes replacement memory and advances the processed position. Failed or
cancelled excerpts leave progress unchanged; source edits revoke partial state.
Text folding uses the same custom compression builder and parser as whole
batches. Raw audio is not split by the text algorithm.

Visual input uses existing `input_image` items with JPEG/PNG data URLs, including
client-extracted video frames. No video-file decoder or audio-append frame path
is added. Completed visual sources become eligible when selected dialogue needs
compression, when they leave the recent-message target, or when unsummarized
images reach the per-request image limit. The runtime processes up to
`visual_chunk_images` (default 8) at a time, shrinking to the actual processor
capacity. Five intermediate summaries (`visual_mid_group`) trigger a long-term
merge; at most 15 long-term summaries (`visual_max_long`) are retained. Each
visual output is limited by `summary_tokens`, default 512. Oldest long-term
summaries are retired beyond this bound. Source message positions label visual
evidence; they are not video timestamps. Qwen installs visual limits when the
Session model state is constructed; planning does not rewrite live limits.
Both the history state and its deployment limits are required constructor
arguments, so callers cannot silently fall back to default visual limits.

An admitted JPEG/PNG can have a valid header but an undecodable body. For completed
history, the preparation worker identifies that source as unavailable. After
rechecking its identity on the owner loop, Qwen records a visual omission, advances
past it and continues with valid images or dialogue. The failed image is not
included in a visual summary's source dependencies. Resource errors, including
Pillow decompression-bomb protection, and programming errors still fail
preparation without advancing visual progress. Original Session records remain intact, and an undecodable
current input still fails foreground preparation rather than being silently omitted.

Visual summaries are separate from dialogue memory. Source images remain
available until a valid visual result commits; dialogue compression waits for
its visual sources to be summarized or explicitly marked unavailable. Foreground requests include visual memories
and uncompressed images. Consumed originals use a neutral “original image omitted”
marker, which remains accurate if visual memory is retired or excluded from this
request. If current input plus memories exceeds capacity, the
runtime removes dialogue memory and then oldest visual memories before failing
the current request. It subsequently adds older whole turns while they fit.
Image edits or deletions revoke derived visual memory and stale results. Worker
cancellation joins decoding/rendering before another preparation can own its
materials.

The plugin separately provides Session image admission limits. `full_history`,
`bounded_recent` and `incremental` default to 200 stored images / 32 MiB of encoded
image URLs. Explicit `recent` retains the original Session defaults of 8 images /
4 MiB, named
`DEFAULT_IMAGE_HISTORY_MAX_ITEMS` and `DEFAULT_IMAGE_HISTORY_MAX_BYTES` in
`engine/duplex/config.py`. These records are not deleted by summarization.
Reaching this limit rejects additional images; deleting source items also revokes
memory derived from them. The visual preparation store has its own configurable
image count, encoded PNG byte and decoded pixel byte bounds. These are not
process RSS limits.

`bounded_recent` and `incremental` accept text/audio/images and reject tools.
`full_history` and `recent` retain the original image/tool rendering path.

## Architecture

```text
              Python users                                WebSocket clients
        ┌──────────────────────┐                    ┌────────────────────────┐
        │  InlineDuplexClient  │                    │      DuplexClient      │
        │  (DuplexClientBase)  │                    │   (DuplexClientBase)   │
        └──────────┬───────────┘                    └───────────┬────────────┘
                   │   DuplexSessionHandle                      │ /v1/realtime?duplex=1 (alias /v1/duplex)
                   │                                            ▼
                   │                          ┌──────────────────────────────────────┐
                   │                          │ OmniDuplexSessionHandler (thin)      │
                   │                          │  websocket I/O, command_from_realtime│
                   │                          │  event.to_realtime(), attachment /   │
                   │                          │  resume tokens / replay journal      │
                   │                          └──────────────────┬───────────────────┘
                   ▼                                             ▼   DuplexSessionHandle
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ DuplexOmni(AsyncOmni)                      [entrypoints/duplex_omni.py, thin] │
        │   open_session / get_session / resume_session / detach_session / close_session│
        │   DuplexSessionHandle: submit(DuplexCommand) / events() -> DuplexEvent       │
        └──────────────────────────────────────┬───────────────────────────────────────┘
                                               │ engine.open/close/resume/touch_session_async (RPC)
                                               │ engine.submit_command_async (one-way)
                                               ▼
        ┌──────────────────────────────────────────────────────────────────────────────┐
        │ DuplexOmniEngine(AsyncOmniEngine) [engine/duplex_omni_engine.py, thin]        │
        │   loads the plugin, validates session_mode, builds queue messages            │
        │  └─ DuplexOrchestrator(Orchestrator)            [engine/duplex_orchestrator.py]│
        │       template seams + DuplexStagePort over the shared stage machinery       │
        │       ├─ DuplexSessionManager      admission, backpressure, lease reaper,     │
        │       │                            open/close/resume/touch, command dispatch  │
        │       └─ DuplexSessionRunner ×N    ONE session state (DuplexEngineSession),   │
        │                                    ordered mailbox, append planning + stage   │
        │                                    submit, output projection, response /     │
        │                                    playback / history, VAD, typed events     │
        │       plugin: DuplexModelPlugin    (PipelineConfig.duplex_plugin, one/model)  │
        └──────────────────────────────────────────────────────────────────────────────┘
```

In each layer the duplex class extends the turn-based one, which extends
the shared base. The dependency runs duplex -> turn-based and never the
reverse, so the turn-based stack is unchanged and the duplex stack reuses
its request machinery instead of duplicating it:

```text
OmniBase                          OmniEngineBase                        OrchestratorBase
 ├─ Omni       (sync, turn)        └─ AsyncOmniEngine  (turn requests)   └─ Orchestrator        (turn-based admission)
 └─ AsyncOmniBase                      └─ DuplexOmniEngine (session msgs)    └─ DuplexOrchestrator  (hosts DuplexSessionManager,
     └─ AsyncOmni  (EngineClient)                                                                     implements DuplexStagePort)
         └─ DuplexOmni

AsyncOmni._create_engine  -> AsyncOmniEngine     AsyncOmniEngine._create_orchestrator  -> Orchestrator
DuplexOmni._create_engine -> DuplexOmniEngine    DuplexOmniEngine._create_orchestrator -> DuplexOrchestrator
```

Each concrete class constructs its collaborator in an explicit factory
method the base calls at the right moment: `OmniBase.__init__` calls
`self._create_engine(...)`; `OmniEngineBase._bootstrap_orchestrator` calls
`self._create_orchestrator(...)` on the orchestrator thread once the stages
are initialized. `DuplexOmniEngine._create_orchestrator` passes the loaded
plugin and the session runtime config to `DuplexOrchestrator` directly.

### Design rules

1. **One session authority.** The whole duplex session (config, epoch/turn,
   ledgers, lease, stage request resources, model state, projection state) is
   one object, `DuplexEngineSession`, owned by one `DuplexSessionRunner` on
   the orchestrator loop. Nothing above the engine keeps session state beyond
   a handle (session id, event queue, capabilities, public session object).
2. **Typed contract.** Commands into a session are `DuplexCommand`
   dataclasses (`engine/duplex/commands.py`); outputs are `DuplexEvent`
   dataclasses (`engine/duplex/events.py`). `command_from_realtime()` and
   `DuplexEvent.to_realtime()` derive the OpenAI Realtime JSON, so the wire
   format is never hand-built, and the websocket handler and the inline client
   share one conversion.
3. **Generic bases and the turn-based classes have zero duplex vocabulary.**
   `OmniBase`, `AsyncOmniBase`, `AsyncOmni`, `OmniEngineBase`,
   `AsyncOmniEngine`, `OrchestratorBase` and `Orchestrator` carry only
   template seams; `tests/engine/test_duplex_import_boundary.py` checks that
   importing the turn-based stack loads no duplex module.
4. **Deployment selects the serving stack.**
   `vllm-omni serve` constructs `DuplexOmni` when the pipeline declares
   `duplex_plugin` and the deploy configuration sets `session_mode: duplex`.
   An explicit `session_mode: turn` selects `AsyncOmni` instead, without
   changing the pipeline's duplex capability. In duplex mode the server exposes `/v1/realtime?duplex=1` (alias
   `/v1/duplex`), `POST /v1/chat/completions`, `/v1/models` and `/health`, and
   every other turn-based route reports "not available". The chat route uses
   the ordinary chat service on the duplex engine, without a session, when
   the plugin declares `DuplexCapabilities.supports_chat_completions`.
   Turn-based deployments use the ordinary API initialization and retain
   model capability checks and endpoint restrictions.
5. **Serving is transport only.** The websocket handler does socket I/O,
   wire-envelope validation, command translation, event rendering and the
   attachment/resume/replay bookkeeping; it holds no session state.
6. **No `typing.Protocol`** in the duplex surfaces: the plugin, data plane,
   session state, PCM buffer, stage port and client transport seams are ABCs.
7. **The wire codec is not duplex, and a wire type is not a mailbox message.**
   The event and command *vocabulary* plus its wire rendering, the audio format
   negotiation, the conversation-item rules and the error envelope are model-
   and runtime-agnostic, and split along the three tiers
   [`docs/serving/realtime_duplex_api.md`](../serving/realtime_duplex_api.md)
   already defines: **Tier 1** (identical to OpenAI) is
   `vllm_omni/protocol/realtime/**`; **Tier 2** (OpenAI names carrying our
   extensions) and **Tier 3** (ours alone) are `vllm_omni/protocol/duplex/**`.
   A Tier 2 class *subclasses* its Tier 1 twin and declares only what it adds,
   so the OpenAI surface stays honestly pure --- `session.created` without
   `resume_token`, `input_audio_buffer.append` without `video_frames`. Our
   error **codes** are Tier 3 (`protocol/duplex/errors.py`); only the envelope
   shape and OpenAI's three `error.type` classes are Tier 1.
   What stays engine-side is everything that is *internal representation*
   rather than contract: `DuplexCommand.payload()` renders the session runner's
   mailbox dictionary, whose channel genuinely differs from the client event
   (`session.update` and the three `conversation.item.*` commands all travel on
   `turn.signal`); `engine/duplex/realtime_commands.py` decides which command a
   decoded event becomes; `engine/duplex/realtime_events.py` holds the
   session's projection state. The codec may not import the engine, the
   entrypoints, `model_executor` or the clients
   (`tests/protocol/realtime/test_protocol_import_boundary.py`), and there is
   exactly one implementation of each codec behaviour
   (`tests/protocol/realtime/test_realtime_codec_single_source.py`).
   The dependency chain is `protocol/realtime` <- `protocol/duplex` <- engine /
   entrypoints / clients, and a duplex consumer uses **only** the middle link:
   `protocol/duplex/**` re-exports the Tier 1 names it does not extend, so a
   helper that later needs a duplex-specific version (`convert_input_audio_with_rate`
   resamples to MiniCPM-o's 16 kHz rather than the client's rate) is overridden
   in one file instead of at every call site
   (`tests/protocol/duplex/test_duplex_protocol_facade.py`).
   `DuplexEvent` is therefore a plain alias of `RealtimeEvent` --- an event has
   no engine-internal half --- while `DuplexCommand` is a real class, because a
   command does.

## Package layout

```text
vllm_omni/
├── entrypoints/
│   ├── omni_base.py                 OmniBase (+ abstract _create_engine)
│   ├── async_omni_base.py           AsyncOmniBase (output pump, _route_engine_message seam)
│   ├── async_omni.py                AsyncOmni (turn-based requests)
│   ├── duplex_omni.py               DuplexOmni, DuplexSessionHandle (thin pipe)
│   ├── duplex/                      SERVING (thin)
│   │   ├── serving.py               OmniDuplexSessionHandler
│   │   ├── realtime_input.py        RealtimeEnvelope (query rules, first message), parse_resume_request
│   │   ├── session_attachment.py    DuplexSessionAttachmentRegistry (resume tokens, replay journal)
│   │   ├── audio_encoding.py        encode_audio, injected into DuplexOmniEngine for the plugin's data plane
│   │   ├── openai.py                dispatch_realtime_websocket (turn-based /v1/realtime, when duplex mode is not selected)
│   │   ├── warmup.py                startup warmup probe; holds native session admission after WebSocket acceptance
│   │   └── websocket.py             websocket send/close/receive helpers
│   └── openai/api_server.py         builds DuplexOmni for duplex models; session-backed app state
├── protocol/                        SHARED WIRE CODEC (no engine / no model / no transport)
│   ├── realtime/                    the OpenAI Realtime API itself
│   │   ├── events.py                RealtimeEvent + OpenAI's 30 server events (20 ship as Tier 1;
│   │   │                            10 have a Tier 2 twin in protocol/duplex)
│   │   ├── commands.py              RealtimeCommand + OpenAI's 10 client commands (8 Tier 1,
│   │   │                            2 with a Tier 2 twin) --- wire_type + fields, no payload()
│   │   ├── formats.py               audio format spellings, what is supported, per-session validation
│   │   ├── session.py               session-object readers; RealtimeInputDefaults (append defaults)
│   │   ├── items.py                 conversation item shape, truncation, transcripts, camera frames
│   │   ├── audio_input.py           decode_audio_append -> RealtimeAudioAppend; client speech hints
│   │   ├── audio.py                 PCM / G.711 / WAV decode, resample, re-encode
│   │   ├── errors.py                RealtimeProtocolError (the envelope's exception type)
│   │   └── capabilities.py          RealtimeProtocolCapabilities: what one consumer can serve
│   └── duplex/                      Tier 2 + Tier 3, and the one door a duplex consumer uses
│       ├── __init__.py              re-exports the Tier 1 helper functions (audio, formats,
│       │                            items, session, capabilities) so the engine never
│       │                            imports protocol/realtime directly
│       ├── events.py                the whole 44-event vocabulary: 20 Tier-1 re-exports,
│       │                            10 Tier-2 subclasses (session.created + resume_token, ...),
│       │                            14 Tier-3 events (listen/speak, playback, resume, ...);
│       │                            DuplexEvent = RealtimeEvent
│       ├── commands.py              the whole 17-command vocabulary: 8 Tier-1 re-exports,
│       │                            2 Tier-2 subclasses (append + hints, commit + final),
│       │                            7 Tier-3 commands (barge_in, playback.ack, ...)
│       └── errors.py                our error-code vocabulary (Tier 3) + RealtimeProtocolError
├── engine/
│   ├── omni_engine_base.py          OmniEngineBase (stage processes, orchestrator thread, queues, RPC)
│   ├── async_omni_engine.py         AsyncOmniEngine (turn-based request building)
│   ├── duplex_omni_engine.py        DuplexOmniEngine (session message surface, creates DuplexOrchestrator)
│   ├── orchestrator.py              OrchestratorBase + Orchestrator
│   ├── duplex_orchestrator.py       DuplexOrchestrator (+ DuplexOrchestratorRequestState; implements DuplexStagePort)
│   └── duplex/
│       ├── commands.py              mailbox half: DuplexCommand.payload() + the mailbox `type`,
│       │                            paired with each protocol command; command_from_realtime
│       ├── realtime_commands.py     duplex binding of the codec: decoded event -> DuplexCommand,
│       │                            DUPLEX_REALTIME_CAPABILITIES, duplex_response_format
│       ├── events.py                re-export shim (DuplexEvent = RealtimeEvent) + the runner's
│       │                            epoch-filter sets DOMAIN_TERMINAL_EVENTS / MODEL_OUTPUT_EVENTS
│       ├── delivery.py              DuplexOutputBuffer (bounded output shared with the session handle)
│       ├── realtime_events.py       RealtimeProjectionState: internal event -> typed events
│       ├── messages.py              queue envelopes (Open/Close/Resume/Touch/Command/Result/Event), DuplexSessionError
│       ├── config.py                DuplexSessionConfig, DuplexCapabilities, ResponseCreateOptions
│       ├── contracts.py             DuplexFence (session_id, epoch, turn_id), stage request records, DuplexStagePort
│       ├── plugin.py                DuplexModelPlugin, DuplexModelSessionState, DuplexDataPlane, PcmAppendBuffer ABCs
│       ├── turn_detection.py        server-side VAD turn detector used by the session
│       ├── audio.py                 compatibility re-export, via protocol/duplex
│       ├── vad.py / intermediate.py
│       ├── response_judge.py        post-ASR judge: does a committed turn need a reply
│       └── session/                 one engine-resident session and everything that runs it
│           ├── engine_session.py    DuplexEngineSession: ledgers, lease, fence, stage resources, append sequencing
│           ├── runner.py            DuplexSessionRunner (per-session mailbox on the orchestrator loop)
│           ├── manager.py           DuplexSessionManager (admission, backpressure, reaper, dispatch)
│           ├── context.py           DuplexSessionContext / DuplexRunState: what the runner shares with its components
│           ├── emitter.py           SessionEmitter: projection, epoch filter, domain effects of a terminal event
│           ├── model_channel.py     ModelChannel: submit an append, project stage output, continue a turn
│           ├── memory_runtime.py    SessionMemoryRuntime: execute private model-planned memory requests
│           ├── control.py           SessionControl: server VAD and the events that reconfigure it
│           ├── append_task.py       AppendAttempt: one append in flight and the rollback its failure owes
│           ├── input_clock.py       InputClock: model units and input acknowledgements of input-clocked sessions
│           ├── helpers.py           pure reads and payload builders over a session
│           ├── lease.py             DuplexLeaseState (idle TTL, disconnect grace, resume generation)
│           └── overlap_policy.py / commit_policy.py / playback_ledger.py
├── config/stage_config.py           PipelineConfig.duplex_plugin; DuplexSessionRuntimeConfig
├── model_executor/models/minicpmo_4_5/duplex/plugin.py   MiniCPMO45DuplexPlugin (+ data_plane, input, policy, ...)
├── model_executor/models/nemotron_voicechat/duplex/plugin.py   NemotronVoiceChatDuplexPlugin (+ data_plane, input, capabilities)
├── model_executor/models/personaplex/duplex/plugin.py   PersonaPlexDuplexPlugin (+ stage0, data_plane, input, policy)
├── model_executor/models/qwen3_omni/duplex/plugin.py   Qwen3OmniDuplexPlugin (+ input, session)
├── model_executor/models/aura_omni/duplex/plugin.py   AuraDuplexPlugin
└── clients/
    ├── duplex.py                    DuplexClientBase (ABC), DuplexClient (websocket), client-side events
    ├── inline_duplex.py             InlineDuplexClient (in-process, over DuplexOmni)
    ├── minicpmo_4_5.py              MiniCPM-o 4.5 session preset
    ├── nemotron_voicechat.py        Nemotron VoiceChat session preset
    └── personaplex.py               PersonaPlex session preset
```

## Session identity

- **The server allocates every session id.** `DuplexOmni.open_session`
  generates `duplex-<uuid4 hex>`; a `session_id` / `id` key inside the
  Realtime session object is ignored (Realtime clients echo the session object
  back). The allocated id reaches the client in `session.created` and is the
  only handle for `session.resume`, `close` and every command.
- **Ids are never reused** within an engine's lifetime, so the id alone
  identifies a session. There is no incarnation counter: stage request ids are
  `duplex-s.<b64 session id>.e.<epoch>.r.<role>`, model-side per-session state
  is keyed by `session_id` alone, and a command, close, touch or resume for an
  id the manager no longer holds is answered with `unknown_session`.
- **Staleness inside a live session is epoch-based.** `DuplexFence
  (session_id, epoch, turn_id)` is an engine-internal identity: every stage
  request is bound to the fence it was submitted under, cancels advance the
  epoch, and tracked tasks re-check the epoch after each `await`.
- **Resume identity** is `(session_id, resume_token, expected_lease_generation)`:
  the token belongs to the attachment registry, the lease generation is the
  engine-side compare-and-swap.

### Authoritative history snapshots

`DuplexEngineSession.history_snapshot()` synchronously captures a
`DuplexHistorySnapshot` on the session owner's loop, without an `await`.
Its aligned tuple columns are `messages`, public `item_ids` (possibly `None`),
internal `source_ids`, and `source_revisions`, together with one
`history_revision`. Unresolved history placeholders are omitted from every
column. The record is frozen; nested protocol dictionaries and lists are
detached, mutable copies for plugin compatibility. Text, bytes and opaque
media payloads are shared, rather than cloned. Consumers must treat opaque
payloads as read-only. Session mutation methods own authoritative writes.

Every non-placeholder history insertion must call `_track_history_message`,
and removal must call `_forget_history_message`, keeping the source map complete
for snapshot capture and hard-cutoff updates. Hidden placeholders are untracked.

The default prompt path uses `session.history_for_prompt()` to capture aligned
history and item IDs while preserving Qwen retained-PCM identity. Like
`session.history` and unlike `snapshot.messages`, it copies only top-level
message dictionaries; nested
protocol/media containers and payloads remain shared with authoritative history
and must be treated as read-only by prompt consumers.

A source ID is allocated monotonically when a message becomes visible and
is never reused within that session. Replacing a public item's contents
preserves its source ID. Deleting it retires the source; recreating the same
public item ID allocates a new source ID. A late item-ID binding preserves
the source ID and advances its revision. New sources start at revision zero.

Content edits, accepted changes to heard-prefix evidence, and tighter
permanent audio cutoffs advance the affected source revision and the global
history revision. A cutoff change can advance revisions even when the text
stays equal. Appending or deleting a source advances the global revision.
Equal-value replacements, rejected operations, hidden reservations, and
internal playback/cache cleanup alone do not. Revisions are monotonic change
tokens, not operation counts; one operation can perform multiple writes.

Epoch changes fence in-flight work independently of source validity.
Re-leasing the same live session on resume preserves IDs and revisions;
dropping the session drops this state. Snapshots do not imply persistence
across process restarts. Existing prompt construction remains unchanged;
stage views and Qwen L2 memory consume these snapshots without changing Session
history ownership.

### Derived context and per-stage views

`DuplexEngineSession.context_ledger` holds derived nodes. A node records the
`(source_id, revision)` pairs it was built from and the IDs of its parent
nodes; parents always have smaller IDs. Nodes never write to history or
advance a revision, and node IDs are never reused.

A node is valid only while every recorded source has the recorded revision in
the current snapshot and every parent is valid. `refresh(snapshot)` walks the
nodes in ID order and prunes the invalid ones, so a content edit, a transcript
update, a deleted source, an accepted heard prefix or a tighter hard cutoff
removes the affected nodes and all of their descendants. An epoch change alone
removes nothing. `add_node` admits a node only if its sources and parents are
valid against the snapshot it is given, so a result computed from older
sources is rejected when checked against a current snapshot. Admission with
a historical snapshot does not displace a newer L1 source lookup; duplicate
L1 nodes for the same source revision are rejected through either admission
API. Callers refresh against a current authoritative snapshot before reading
valid nodes.

`get(node_id)` reads a node after refresh. `remove(*node_ids)` explicitly retires
nodes and their descendants, preserving unrelated nodes and L1 lookup indexes.
`replace_summary(snapshot, sources=..., replaces=..., message=...)` validates new
source references and all replaced summaries against the supplied snapshot. It
copies the complete transitive source dependencies into the new `SUMMARY`, then
removes the old summaries and their descendants. The new node has no parent edges,
so retiring old summaries does not invalidate their replacement. Invalid sources,
missing replacements, non-summary replacements or message-copy failures leave the
ledger unchanged. This API does not refresh implicitly: the caller supplies the
current authoritative snapshot for result acceptance.

Admission detaches message dictionaries, lists and tuples from the caller while
sharing opaque payloads. Batch admission prepares every message before writing
nodes, so a preparation failure leaves the graph and next node ID unchanged.
The ledger owns admitted message containers; node readers must treat them and
opaque payloads as read-only. Rendered views are detached from ledger messages.

A plugin opts a stage in by returning a `DuplexContextViewPolicy` from
`context_view_policy(stage_id=..., runtime_config=...)`. The default is `None`:
the prompt config is built exactly as before and no L1 view is rendered. With a
policy, `conversation` stays the authoritative history and the stage also
receives `conversation_view`, a list aligned one-to-one with `conversation`:

- User messages before the `keep_media_user_turns` most recent user runs
  (consecutive user messages) drop their `audio_url` and `image_url` parts and
  keep their text. The transcript is added as a text part when no text part
  already carries it.
- A user run is never emptied. If every message in a run would be empty, the
  whole run is passed through unchanged.
- Assistant, system and tool messages are never rewritten. Assistant text
  already reflects the accepted heard prefix.

This is a candidate context view, not a complete prompt: with a zero-keep
policy, unanswered input is eligible too. Plugins must preserve current
request media from `conversation`, and keep binding audio and validating
sources against it. ModelChannel reuses the snapshot captured for item IDs
when rendering the view, so both come from the same capture. Policy config
is deployment-owned; plugin request runtime state is a separate input.

Qwen3-Omni opts in its Thinker stage when
`duplex_session.context_keep_media_turns` is set (default `null`). Its default
`recent` policy formats older user items from the view, including mixed
text/media items. The experimental Qwen history policies instead select original
turns and ledger summaries using their own context and media budgets; the L1
candidate view does not override that selection.
Dropped images are never decoded, and a text-only view replaces old audio
even while its payload remains in the retained audio slots. A turn whose
audio left those slots keeps its available transcript instead of being
dropped. The current unanswered user input remains verbatim, including during
tool continuations, after intervening system instructions, and for audio and
image items when the knob is `0`.
The 16-message window, the
audio-slot limits and the no-empty-run fallback are unchanged.

## Session lifecycle

```text
open   DuplexOmni.open_session(config)
         -> engine.open_session_async(session_id, DuplexSessionConfig)      [RPC]
         -> DuplexSessionManager.open: admission (max_sessions, closing sessions keep their slot),
            plugin.validate_client_extra_body / prepare_runtime_config, DuplexEngineSession,
            Stage0 request id reserved (ensure_stage_request), DuplexSessionRunner.start()
         -> first event: SessionCreated (announces the allocated id)
command  handle.submit(DuplexCommand)
         -> engine.submit_command_async -> DuplexSessionCommandMessage on the request queue [one-way]
         -> DuplexSessionManager.dispatch: unknown_session / input_backpressure checks, then runner mailbox
output   DuplexOrchestrator._intercept_stage_output -> runner.on_stage_output -> mailbox -> typed events
         -> session's bounded output buffer -> handle.events(); closure and errors without a live session
            use output_sink (DuplexSessionEventMessage) -> DuplexOmni._route_engine_message
detach   DuplexOmni.detach_session(expected_lease_generation) -> touch(DETACH): engine-owned disconnect
         grace, refused for a lease newer than the one the caller held; expiry -> SessionExpired
resume   DuplexOmni.resume_session(expected_lease_generation) -> lease CAS keyed by the control id (a replay
         answers with the generation it produced); the existing handle is re-entered. A caller cancelled
         mid-RPC observes the outcome afterwards (replaying until the engine answers) and settles a landed
         resume: the generation goes to the resume waiting to activate, else to the connection still
         attached, else the lease is detached again
close    DuplexOmni.close_session -> close RPC; the manager tears the runner down, then the stage cleanup
         (abort submitted requests, release reserved ids), then SessionClosed, then the RPC result.
         SessionClosed is emitted after the cleanup attempt (in a finally), so it is also sent when the
         stage cleanup failed; in that case the admission slot is intentionally retained and the cleanup
         is retried by the reaper, and opening a replacement session can still be refused with
         resource_exhausted. Seeing the event therefore does not by itself guarantee a free slot.
reap     DuplexSessionManager.reaper_loop: idle TTL / disconnect grace expiry, cleanup retries
```

### Concurrency and ordering model of the runner

The runner's session state is owned by the orchestrator asyncio loop and
needs no lock. The output buffer shared with the caller thread uses a lock
for queue updates and the final validity check before delivery.

- **Single writer per session.** `DuplexEngineSession` is mutated only by its
  runner. Inputs reach the runner through one mailbox in this order: client
  commands (from `_request_handler`, in request-queue order), stage outputs
  (`on_stage_output` enqueues an internal item; it does not mutate the session
  inline), timers (silence continuation, bounded auto-response, the input
  clock's timeout check) and internal items (deferred overlap promotion,
  VAD-synthesized commits; in an input-clocked session also the
  `StageProgress` record queued behind each stage output, the marker of an
  input refused at admission, and the settling of an append task). The
  mailbox worker processes items one at a time.
- **Slow work is not awaited inline.** An `AppendAudio` handler reserves
  bytes, then schedules a tracked append task on the per-session append tail:
  the task awaits `_offload(decode/resample/VAD)` on a dedicated
  `ThreadPoolExecutor` owned by the manager, then `plugin.plan_append`, then
  `stage_port.submit`. The worker returns to the mailbox immediately, so a
  later `CancelResponse` is not blocked behind an in-flight append.
- **Re-validation after every `await`.** A tracked task captures `(epoch,
  turn_id)` when it starts and re-checks them before the stage submit and
  before committing ledgers; a task that finds the epoch advanced rolls back
  its PCM reservation and exits.
- **Buffered commit prefixes.** A model buffer may reserve non-terminal units
  before its terminal commit payload. The runner submits these in wire order
  with `force_listen=True` and waits for append acceptance before acknowledging
  the commit. A commit that must wait for an active response holds those
  prefixes and submits them, still in wire order, when that response ends,
  before the retained terminal payload. Reservations keep their normal rollback
  ownership. MiniCPM uses this path to preserve separate buffered VISION units
  and a following speech tail in manual commit/response sessions. Native
  resumable responses also defer video-only commits. If response creation is
  disabled on the commit, a later explicit `response.create` must drain the
  held prefixes before submitting the terminal payload.
- **Coalesced camera commits.** Single-terminal models such as JoyAI combine
  frames from retained camera commits in capture order. The newest commit
  owns the response while earlier committed frames remain available to model
  memory. Retained frames continue to consume the pending-input byte budget
  until submission or cleanup. Before commit, the model's frame buffer applies
  its sampling policy, such as retaining only the latest camera frame.
- **Cancel is atomic at the session.** `CancelResponse` / `BargeIn` advance
  the epoch, cancel tracked append tasks, abort the owned stage bindings
  through the stage port and emit the cancelled terminal events before
  yielding. An append task already past its last check is harmless: its stage
  request is bound to the old epoch's request id, which the abort covers, and
  the base never emits session-owned outputs to a client.
- **Tool calls settle once, and the epoch settles them.** A function call the
  runtime exposes to the client (`response.function_call_arguments.done`) is
  recorded in a session-owned ledger before it is emitted, and its
  `function_call_output` is admitted only against that record:
  `pending -> accepted | timed_out | cancelled`, one transition, no rebinding
  of a call id, and terminal records kept as tombstones until the session
  closes. A client can also settle one call with `function_call.cancel`. Conversation history is not admission authority -- it is history-wide
  and epoch-blind, so a result arriving after a barge-in could otherwise be
  admitted against a call the session had already moved past. The epoch
  consequences hang off the single epoch-advance path: pending calls from older
  epochs are cancelled (or, with `function_call_interrupt_policy: keep`, move to the new
  epoch and stay answerable), and a result the model accepted but never consumed is
  retired from the runtime so the new epoch cannot speak it. A retired or
  already-consumed generation stays *named* in the worker-facing runtime config
  (model-owned state is per resumable stage request, and a cancel moves the
  session to a different one), so the next request starts above what the
  previous one took instead of re-speaking it or rejecting the next result as
  missing; a generation that only *looks* missing is still refused by the
  worker. Eligibility is the
  origin epoch plus the settlement, never the current turn: a model turn
  completing while the client answers does not invalidate the result, and the
  continuation it triggers targets the turn that is current at admission.
  An optional deadline (`extra_body.tool_result_timeout_s`, off by default)
  times a pending call out; its timer callback only enqueues the call identity
  onto the session's mailbox, so the deadline decision is serialized with every
  other command instead of running inside an `asyncio` timer callback.
- **Ordered event stream.** Because the epoch bump and every emission happen
  on one loop in program order, nothing can be emitted for an old epoch after
  its terminal event; `runner.emit()` still drops a terminal carrying a stale
  epoch and stamps `epoch` on every event for clients that filter
  defensively. In this experimental follow-up, public output goes directly to
  one bounded buffer shared by the engine and `DuplexSessionHandle`. Accepted
  cancellation removes queued `AudioDelta` events for that response through
  the cancelled epoch. Text increments, audio/transcript completion markers,
  other responses and normal completion retain FIFO order. The consumer also
  checks an event already taken from the buffer: WebSocket delivery repeats
  that check under the connection lock, before sequencing and journaling.
  Already sequenced replay entries are unchanged; already delivered audio
  still requires the client to stop its own playback.
- **Backpressure before the mailbox.** `DuplexSessionManager.dispatch`
  checks `max_pending_input_bytes_per_session` and reserves a pending turn for
  `Commit` before the put; a rejected command is answered with
  `ErrorEvent(code="input_backpressure")`.
- **Per-response stage metrics.** `_route_output` computes the per-segment
  `StageMetrics` before calling `_intercept_stage_output`; the runner
  accumulates the snapshot into the active response, so
  `ResponseDone.stage_metrics` and `metadata.vllm_omni.stage_metrics` on the
  wire keep their meaning.
- **Session stats (busy RTF, RFC #8325).** Opt-in with
  `duplex_session.collect_stats`. `ModelChannel` stamps each Stage-0 append's
  envelope with `duplex.perf_chunk = [epoch, seq]` (a seq retried after a failed
  submit gets a fresh id, so the aborted submit's late spans are ignored); the orchestrator copies the
  producing step's chunk into forwarded updates and the chunk transfer adapter
  into connector payloads. Stage schedulers (`DuplexStageWorkRecorder`) time
  every step of a request that carries a chunk and return the spans on
  `OmniEngineCoreOutput.stage_work`, which the orchestrator hands to the
  session as it collects them. `DuplexEngineSession.perf`
  (`SessionPerfLedger`) groups chunks by turn with their acceptance times and
  spans (a chunk is done at its last span end), and records each turn's emitted audio through the
  response it belongs to; turns close on model-turn completion, barge-in or
  close. The runner answers `session.stats.get` from the ledger on
  the ordered mailbox; clients derive busy RTF and stall ratio
  (`vllm_omni.benchmarks.duplex_session_metrics`). A turn is reported final
  only after the model turn completed, its responses ended and every chunk was
  handled; cancelled turns never become final, and later spans are counted in
  `late_spans`. With the flag off nothing is stamped or attached, and stage
  schedulers check only newly scheduled requests. Single-host only: the API and
  stage processes must share one `CLOCK_MONOTONIC`.

Model plugins may opt into one bounded auxiliary request per Session. The
Session-owned `SessionMemoryRuntime` gives it a separate resource identity,
submits and terminates it at the selected stage, and intercepts its output and
errors before response policy, speech routing, playback history or response
metrics. The plugin receives immutable
committed history and owns prompt/result formatting and source validation. Sources
can cover text, images or model-owned derived state. The helper checks the epoch,
session configuration revision, operation identity and plugin source validity before
mailbox writeback. Appending newer history can preserve a stable prefix; editing
or deleting its evidence invalidates it. Cancellation and close abort owned
requests. A failed cleanup retains its resource for teardown retry and prevents
new auxiliary work from accumulating behind it. Each auxiliary cleanup attempt
uses the shorter of five seconds and the model request timeout, keeping long
generation deadlines separate from this cancellation-safe cleanup wait.
The configuration revision tracks adopted Session settings; model-owned chunk
and tool progress retain their existing stage generation and preserve pending
memory work while its source remains valid.
The plugin supplies a stable work key and whether work can enter the existing
stage queue during an ordinary observation. Explicit user interaction takes
priority through cancellation after its append plan passes validation, and
suppresses restart until its turn completes. Rejected input leaves an in-flight
summary running so it can release memory capacity for the next input.
Requests may specify short retry delays and a cooldown for subsequent attempts;
failures remain keyed to the same source as newer observations arrive. Background
failures are logged on the server. Successful application allows the next
bounded chunk to catch up without waiting for new input.
Plugins with `uses_history_context=True` also receive
`bind_history_context(state=..., snapshot=..., ledger=...)` before foreground
preparation, auxiliary planning and auxiliary source validation. The owner first
refreshes the Session ledger; validation and result application run consecutively
on the owner loop without yielding. The hook supplies source authority while
leaving existing model-specific auxiliary request and compression interfaces
unchanged. Plugins that do not opt in retain their existing path.

The committed-history revision lets auxiliary events reuse an unchanged snapshot.
The ledger is still refreshed before use, and Qwen compares public source IDs and
revisions before submission and writeback. These checks can grow with retained
history and summary dependencies; they do not provide constant CPU or memory cost.
The runner delegates auxiliary lifecycle and retry state to this helper through
`DuplexSessionContext`, tracks its tasks, and dispatches completion events through
the existing mailbox. Model plugins retain the existing policy hooks and own
content policy, memory tiers, prompts and budgets. Recoverable input-capacity
rejection releases the unsubmitted request and its reservations while keeping
the Session open.

### Semantic turn decisions

`SessionControl` owns a detector per session and offloads its ordered audio
processing to the manager's executor. `SemanticTurnDetector` loads the shared
Smart Turn backend on the first audio chunk; the backend constructor warms the
CPU ONNX session. Silero's candidate pauses trigger subsequent scoring. Each
detector retains the current pause's semantic decision until speech resumes or
the turn resets. At commit, session-correlated logs distinguish a complete
semantic decision from the silence timeout, including whether the preceding
score was incomplete or inference failed. These diagnostics do not change the
wire protocol or the endpoint decision.

Only unfinished response-bound append tasks keep the response gate closed.
A completed task awaiting its cleanup callback does not block a new turn after
barge-in. Active responses and unplayed output retain their own gates.

### Experimental output limits

`max_pending_output_bytes_per_session` defaults to 2 MiB and
`max_pending_output_events_per_session` to 512. The byte limit counts compact
Realtime JSON, including base64 audio, not decoded PCM or Python object
overhead. Audio is already base64-encoded by the producer; counting its length
avoids serializing or decoding that large field again. Neither default has
been tuned from a performance study.

Errors and response endings have an additional 64 KiB / eight-event reserve
and retain FIFO order. One final session-closure notification has an independent
slot, so an exhausted reserve cannot prevent closure or break delivery to
other sessions. If that reserve also fills, an error or response-ending
notification may be omitted; the final session-closure notification still
has its independent slot. Oversized close details (over 4 KiB with generated identity)
are replaced by a compact reason, preserving terminal type and identity.
Late writes after closure are ignored; a locally ended iterator is not reopened
by a late terminal, and a failed open closes the abandoned handle. Events other
than errors are not forwarded once their session's buffer has been removed.
One event held by the consumer is outside the queued budget.

The overflow policy fails the active response without committing its partial
text to history and closes only the affected session. If overflow occurs while
a completion batch is being delivered, its not-yet-queued `ResponseDone` is
converted to failure with its identity and output preserved; any history entry
for that response is removed. An ending already accepted by the buffer is not
replaced or followed by a second ending. After overflow, newly
emitted ordinary events are suppressed, including text/audio completion,
content-part and output-item completion markers. Previously queued non-audio
events retain FIFO order; error and failed response-ending notifications may
use the reserve before the independent closure notification. This differs from
accepted cancellation, which removes only matching audio. See the
[public error and closure fields](../serving/realtime_duplex_api.md#output-limits-and-slow-consumers).
Response-only recovery is not implemented. These limits do not bound the runner's raw stage
output mailbox, replay journal, or client playback queues. The shared buffer
requires the current same-process, separate-thread engine arrangement; it is
not a cross-process delivery protocol. Direct handle iteration and WebSocket
delivery share buffer validity, but end-to-end client behavior still needs
independent real-model verification.

## Session-aware replica placement

The opt-in `duplex_session.replica_policy: least_sessions` policy places each
session on a replica group mapping every stage to a replica, including replicas
attached from headless runtimes on other hosts. The orchestrator owns a separate
session reservation ledger; StagePool's request
bindings continue to describe submitted requests only. Admission reserves a
group synchronously after taking the global admission slot. The group must
have space under `max_sessions_per_replica` on every member replica. Replicas
shared by groups share their capacity counter. Eligible groups are ordered by
maximum reserved-to-capacity ratio, total reserved count, and round-robin
tie-breaking.

The reservation covers opening, active and closing sessions. Failed opens
roll it back; ordinary request completion, response cancellation and epoch
changes do not release it. Session cleanup releases it only after the closing
session is finalized. Every Stage0 submission, downstream forward, prewarm and
derived request uses the session's pinned stage replica. Routing is fixed
before multimodal preprocessing and connector sender information are built.
A conflicting or unavailable pin fails instead of falling back to another
replica.

Replica failure marks the replica unavailable and closes every session in its
reverse index, including idle sessions without an in-flight request. Other
groups retain their reservations and can continue serving. This policy does
not migrate model state. Distributed pins use StagePool's explicit-routing
bindings and hub availability checks; detach notifications also close idle
sessions. Session reservations remain separate from request bindings.
Worker-local state capacity uses the per-replica cap, while `max_sessions`
remains the global service cap. The default `legacy` policy preserves the
existing routing, admission and worker capacity behavior. See the
[configuration reference](../configuration/stage_configs.md#duplex-replica-placement)
for group validation and model capability requirements.

With metrics enabled, `vllm_omni:duplex_reserved_sessions` reports reservations
by model, stage and replica, and `vllm_omni:duplex_admission_rejected_total`
counts global and replica capacity rejections. These collectors are created
on first use by this policy and carry no session ID labels.

### Multiple runtimes and hosts

A runtime is one `vllm serve --omni` invocation. In a stage-based deployment,
each stage-hosting runtime launches local replicas of one stage. One runtime is the head:
it omits `--headless`, serves the API, and owns the master and session
orchestration. Any stage runtime can take this role. The remaining runtimes
use `--headless` and connect to the same `--omni-master-address` and
`--omni-master-port`. A stage can span several runtimes, including runtimes on
different hosts.

For example, the declared stage totals `[1, 3, 1]` can be placed as follows:

| Host | Stage | Local replicas | Role | Runtime arguments |
| --- | --- | --- | --- | --- |
| host0 | 0 | 1 | headless | `--stage-id 0 --omni-dp-size-local 1 --headless` |
| host0 | 1 | 1 | head | `--stage-id 1 --omni-dp-size-local 1` |
| host1 | 1 | 2 | headless | `--stage-id 1 --omni-dp-size-local 2 --headless` |
| host0 | 2 | 1 | headless | `--stage-id 2 --omni-dp-size-local 1 --headless` |

The head's `stages[].num_replicas` declares the cluster total for each stage;
`--omni-dp-size-local` specifies the runtime's local share. For its own stage,
the head resolves the total to `max(configured_total, local_count)`, preserving
the existing CLI override when the configured count is smaller. The first
`local_count` slots launch locally; the remaining slots attach remotely.
Other stages attach remotely in their entirety. The existing unfiltered
`single_stage_mode=True` path keeps all configured replicas local.
The existing frontend-only head keeps all stage replicas remote.

Device pools are split against the runtime's local replica count. For example,
total `3`, local `2`, TP `1`, and `devices: "0,1"` assign GPUs 0/1 to the two
local replicas; the third registration slot has no local device assignment.

`DuplexReplicaPlacement` owns the startup group map and session reservations.
Omitting `replica_groups` generates the Cartesian product of the stage replica
slots, equivalent to a SQL `CROSS JOIN`, for both equal and unequal stage counts.
The example above automatically gets three groups `{0: 0, 1: i, 2: 0}` for
`i = 0, 1, 2`; `[2, 2]` gets all four pairs. The last stage's replica index
changes fastest when assigning group IDs.
Implicit paths are selected lazily from the least occupied replicas at each
stage, preserving the same group IDs and round-robin tie-breaking without
enumerating the product. Diagnostics record stage counts and the total group
count instead of expanding every implicit path. Explicit groups restrict the paths
instead of adding to the generated product. Each session reserves one path;
replicas shared across paths retain one capacity counter. Stage 0 and stage 2
each need a per-replica session cap of at
least three to admit three sessions concurrently. Registration-assigned replica
IDs identify replica slots, not durable host identities; explicit groups do
not establish host affinity.

StagePool owns dispatch and request bindings. A duplex pin reuses
`bind_requested_replica`, including attachment and hub-UP validation, and uses
the existing distributed address-affinity ledger. Internal routing errors are
translated to stage failures. There is no second distributed pin ledger and no
fallback to another replica after a required pin becomes unavailable.
`MembershipController` invalidates the removed address and notifies the
orchestrator. Duplex retirement marks the replica unavailable and closes every
session reserved on it, including sessions with no active request. Generic
request cleanup remains responsible for request bindings.

Group topology is fixed for the head's lifetime. Later attachments do not add
groups or recover closed sessions. This design adds neither elastic placement
nor session migration or automatic restoration of an unavailable group.
The distributed stage topology uses a single API frontend; the existing
`--api-server-count > 1` launcher rejects distributed stage arguments. The
local multi-frontend mode below is a separate deployment mode.

The master address must be reachable from every runtime. Each cross-host
pipeline edge also needs a connector with a suitable remote data plane;
`SharedMemoryConnector` requires its sender and receiver to share a host.
See [Disaggregated Inference](feature/disaggregated_inference.md) for transport
configuration. Control-plane registration alone does not validate a connector's
cross-host data path.

For MiniCPM-o 4.5 on CUDA, keep the deploy profile's native MRv2 runner on all
three stages when selecting `NixlConnector` for remote payload transfer.

### Multiple frontends

`--api-server-count K` reuses the existing local multi-API launcher. The parent
launches all stage cores, each core exposes one IPC endpoint pair per frontend,
and every frontend attaches to every replica. Each frontend has its own
`DuplexOrchestrator`, `DuplexSessionManager` and Python interpreter. Request
`client_index` routes engine output back to the submitting frontend; there is
no central per-frame session dispatcher.

Placement, rather than engine attachment, enforces exclusive ownership. Group
`g` belongs to frontend `g % K`, and only that frontend may reserve its replicas
or submit its session requests. Groups sharing a physical stage replica must
have the same owner, and `K` must not exceed the group count. This validation
also applies to automatically joined groups. Multi-stage joins usually share
replicas across frontend owners, so those deployments must supply explicit
groups with disjoint ownership, such as `[{0: 0, 1: 0}, {0: 1, 1: 1}]` for
two stages with two replicas and two frontends. The global session
budget is divided by floor/remainder across frontends and bounded by each
frontend's owned replica capacity; budgets do not move between frontends.
The reservation lifetime, strict stage pins and worker caps apply within each
owner's groups. With `K=1`, routing and admission behavior are unchanged.

The multi-frontend WebSocket guard allows only `least_sessions` and plugins
without session resume. Credentials, attachment state and replay journals
remain process-local. The shared public address does not provide resume
affinity; supporting resume-capable models would require a separate design.
INET listeners use the existing per-process `SO_REUSEPORT` path where available
(with a shared-socket fallback). Kernel connection distribution is not capacity
aware, so a locally full frontend can return `resource_exhausted` with
`retryable: true` even when another has room. Clients reconnect with backoff;
an established session never moves to another frontend.

When enabled, warmup runs against the local engine and owned groups rather than
connecting through the public listener. PersonaPlex keeps its opt-in behavior:
`duplex_session.warmup_frames` must be positive. Each frontend keeps its own
warmup completion gate; clients wait at most `DUPLEX_WARMUP_CLIENT_WAIT_S` (180 s)
before being admitted, as in single-frontend mode; there is no all-frontend
warmup barrier. `/health`
retains the engine health check, and a 200 response does not guarantee admission
capacity on the frontend accepting a connection. Only the owner publishes its
replicas' reservation gauges, using Prometheus `livesum` aggregation;
admission rejection counters aggregate across
frontends without changing their label schema.

This mode keeps the current supervisor's whole-service fail-fast behavior on
an engine failure or a frontend's abnormal exit. It does not add frontend
restart, session migration or failure isolation. All-frontend engine attachment
and SHM notification fan-out are unchanged; narrowing SHM wakeups is a separate
optimization. See the [configuration reference](../configuration/stage_configs.md#multiple-duplex-frontends)
and [serving guide](../serving/realtime_duplex_api.md#multiple-frontends).

## Orchestrator seams

`OrchestratorBase` keeps the generic machinery (request handler skeleton,
abort, collective RPC, output loops, stage error and dead-replica handling,
`_route_output`, stage forwarding, prewarm, cleanup, shutdown) and offers
these seams. `DuplexOrchestrator` extends `Orchestrator` rather than sitting
beside it: a seam it does not override falls through to the turn-based
implementation, and `_dispatch_message` hands anything that is not a session
message to `super()`:

| Seam | `Orchestrator` | `DuplexOrchestrator` |
| --- | --- | --- |
| `_dispatch_message(msg) -> bool` | `add_request`, `streaming_update`, `add_companion_request`, `interaction` | open/command/close/resume/touch messages -> `DuplexSessionManager` |
| `_background_tasks()` | — | `[manager.reaper_loop()]` |
| `_shutdown_extensions()` | — | `await manager.shutdown()` |
| `_on_stage_submitted(stage_id, request_id, replica_id, req_state)` | — | bind the forwarded stage request to the owning session |
| `_intercept_stage_output(...) -> bool` | — | hand every session-owned output (Stage0 decision, Stage1 audio) plus its metric snapshot to `runner.on_stage_output`; `True` = do not forward |
| `_handle_forward_failure(...) -> bool` | — | fail the owning session's response and close the session, keep the loop alive |
| `_on_replica_removed(stage_id, replica_id)` | — | retire the detached replica and close its reserved sessions, including idle sessions |
| `_cleanup_request_ids(ids, abort=, release_owners=)` | ignores `release_owners` | brackets `super()` with `close_sessions_for_request_ids` / `defer_request_cleanups` / `finalize_closed_sessions` |
| `OrchestratorRequestState.session_owned` | never set | set by `ensure_request`: no synthetic terminal, no client emission, no terminal re-forward, no auto-cleanup on finish |

`DuplexOrchestrator` also implements the `DuplexStagePort` ABC the manager and
runner use (`stage_count`, `sampling_defaults`, `ensure_request`, `submit`,
`cleanup`, `abort_requests`): the resumable Stage0 request is preregistered at
open with `DuplexOrchestratorRequestState(session_owned=True, session_id,
fence)`, submitted with `submit_initial` on the first unit and
`submit_update` afterwards, and the request's `streaming.bridge_states["duplex"]`
carries the session identity (`session_id`, `fence`, `epoch`, `turn_id`,
`model_turn_id`), the public session config and the server-owned runtime
config for the worker-side model hooks.

## Model plugin

`DuplexModelPlugin` (ABC, `engine/duplex/plugin.py`) is the one class a model
provides, selected by `PipelineConfig.duplex_plugin`; "is a duplex model" is
`duplex_plugin is not None`. It merges the engine policy and the session
policy that used to be two separately configured objects:

| Half | Members |
| --- | --- |
| engine policy | `configure_sampling_params(runtime_config, defaults)`, `plan_append(...) -> DuplexAppendPlan` (the resumable Stage0 prompt for one unit), `decide_output(...) -> DuplexOutputDecision \| None` (e.g. the listen decision on a finished Stage0 segment) |
| session policy | `capabilities(max_sessions)`, `validate_client_extra_body`, `prepare_runtime_config(config, model_config)` (server-owned runtime keys, reference audio resolution), `runtime_config_for_update`, `parse_function_call` (recognize one call), `runtime_config_for_function_output` (feed an accepted result back into that model), `note_published_function_call` (plugin stores history length at publish), `has_pending_tool_followup` (plugin still has a tool result to answer), `extra_prompt_config`, `runtime_patch_for_submitted_prompt`, `runtime_config_after_response` (keep or drop a result after the response ends), `runtime_config_after_model_output` (consumption acknowledgement), `create_session_state() -> DuplexModelSessionState`, `data_plane: DuplexDataPlane` (projects raw stage outputs into internal events), `data_plane_context(...)` |

A duplex tool call uses one client sequence. The client supplies tools, receives one function call, runs the function, and sends `function_call_output`. The server does not execute the function. Each model plugin implements both hooks. A recognized call is published through `model_channel._expose_function_call`. Before the plugin sees a `function_call_output`, the session copies `name` and `arguments` from the call origin recorded at publish. Conversation history is not admission authority. The client item stays `call_id` and `output`. Qwen3-Omni `parse_function_call` reads a Hermes `<tool_call>` and returns `{call_id, name, arguments}` without retaining them. One Thinker chunk can carry more than one completed call; each is emitted. `note_published_function_call` stores that call's history length in the Qwen runtime. `runtime_config_for_function_output` reads `name` and `arguments` from the preparation item, refuses when either is absent, and appends one `{history_len, messages}` entry there. The stored length wins over the length at result time. Tool pairs stay out of `session.history`. `has_pending_tool_followup` is how the runner learns that a bare `response.create` may open `{"type":"conversation"}`. `plan_append` splices each turn after the history it was accepted against, and `runtime_patch_for_submitted_prompt` holds those turns aside after the append is accepted. The pair stays owned by that response until playback stores its text. The next prompt puts a still-inflight pair back on the waiting list before the usual splice, and the pair keeps that response id so the playback ACK can drop it once the assistant text is stored. A cancel, a playback clear before any audio is committed, or `realtime_response_conversation: none` stores no text and puts the pair back with that response id. A failure before playback can store the answer puts the pair back without the id, so the retry can own it. `tool_choice: none` is rejected. A result whose published anchor has left the prompt window is refused instead of being attached to the latest question. `runtime_config_after_response` is that step. Nemotron VoiceChat `parse_function_call` reads the data-plane function-channel event, and `runtime_config_for_function_output` places the result on a later frame of the request that is still open. The session call book is #8758.

The application-side `DuplexToolRunner` must consume events before generation
starts. It delivers tool output independently of generation, but only requests
continuation after all observed responses end and ACK-only playback is committed.
Results delivered before the same continuation share one follow-up. Failed result
writes stay visible in `undelivered_results` / `delivery_errors`; `retry_result`
resends the cached output after reconnecting without rerunning the handler.
A failed continuation write instead retains the call in `pending_continuations`
with its `continuation_errors` entry; `retry_continuation` retries only that
follow-up. Continuations reserve awaiting-response state before the send can
yield, so lifecycle events observed during the write remain authoritative.
Only results delivered before a follow-up begins are covered by that request.
An acknowledged hard playback truncate retires the targeted playback gate when
generation ends, including when the ACK precedes terminal metadata. Late ACKs
cannot reopen a retired gate; the early-stop marker is removed at response end.
A successful write is not proof of server acceptance, and the runner never sends
playback ACKs on behalf of the application.

The output-decision and concurrent-turn-release hooks receive the current
request's streaming context. A host plugin can use that context to share a
decision with its stage-input bridge without retaining request state on the
shared plugin. For the optional response judge, AURA returns its listen
decision from `decide_output`; the generic orchestrator does not interpret
judge output. AURA resolves its stage roles in `configure_pipeline`, before
serving requests, and rejects missing required roles or duplicate roles.

`DuplexOmniEngine._validate_deployment` loads the plugin before any stage
starts; `DuplexSessionManager.__init__` validates it against the stage
sampling defaults once the stage pools exist. Plugin hooks that may block
(`prepare_runtime_config` fetching `ref_audio`) are awaited in `open()` and
offloaded from the loop.

Input-clocked sessions (`extra_body.clock == "input"`, see the
[API reference](../serving/realtime_duplex_api.md#input-clocked-sessions))
track every model unit (one Stage0 submission) until all of its output has
been emitted. The clock lives in `session/input_clock.py` and exists only for
sessions created with the option; the runner calls it at explicit points: a
client input is admitted (indexed in wire order when the runner takes it on
its mailbox; an input the manager refuses at admission is indexed by an
explicit `runner.input_refused` call in each refusal branch, which only queues
a payload-free marker acknowledged as `rejected`, like an append the runner
refuses before its audio reaches the model, `_refuse_append`; an append server
VAD split into segments is refused from the refused segment on, the segments
before it stay buffered and its `audio_end_ms` says where they end), begins
and ends, a Stage0 submission is created / accepted / finishes, a committed turn is deferred or
its deferred audio is dropped, a cancel advances the epoch, and teardown. Each
stage output puts a `StageProgress` item on the session mailbox *after* the
events of that output, and an append's outcome and the timeout checks are
settled from the mailbox too, so an acknowledgement never overtakes the output
it covers. A command that waits on the mailbox for an append (a commit or
`session.update`) watches the timeouts of the appends still running and gives
up once one of their units is due or timed out, so a stalled submission
cannot keep the timeout check queued behind it from running. A unit takes its
ordinal among its epoch's Stage0 submissions when Stage0 accepts it, and the
n-th Stage0 segment end of the epoch decides that unit (also after the unit
was settled, so late output is never credited to the next unit; a unit
settled before Stage0 accepted it stays matchable until its submission
resolves, and a deferred turn submitted after its reserved slot was settled
takes that settled unit's place unless a client input submitted it); a
non-final stage only ends a unit through a decision, and only
the final stage completes a speaking unit. Two optional hooks carry what only
the model knows. `unit_decision(stage_id, decision, output, context,
runtime_config) -> DuplexUnitDecision | None` labels a decision of a stage
before the final one and says whether it ends the unit's pipeline (default:
any decision does, labelled `listen` when its metadata sets `model_listen`;
a plugin overrides this when some decisions do not end the unit); it is also
asked about every segment end of such a stage that carries no decision
(default: no classification), so a plugin can end a unit whose segment hands
the next stage nothing. It is called when the output arrives and reads the
live output at that moment. Decisions taken on the final stage are not
consulted. `unit_output_complete(unit, final_stage_id, new_output, new_context,
state, runtime_config) -> bool` says when the oldest speaking unit's
final-stage output is complete; `unit` carries only its session-wide index,
its epoch and its ordinal in that epoch, and two flags (the stage before the
final one ended its segment; the final stage ended its segment for it: the
final stage's segment ends go to the speaking units in order, so a unit that
still waits for its upstream segment end does not take the next unit's), and
the engine keeps no output history (default: the final
stage marks one segment end per unit; a final stage that streams across unit
boundaries keeps its own markers or coverage in `state`, which the engine
replaces with an empty dict whenever a cancel starts a new epoch, so frame
coverage is counted per epoch against the ordinal). Later-stage progress
that overtakes the Stage0 segment end of its unit (no unit is speaking yet,
or none still needs that stage's segment end) is held by the clock and
replayed in arrival order when the next unit of the epoch starts speaking.
It is held only while the epoch has a Stage0 submission without a segment
end yet (the unit it can belong to); otherwise, and once the epoch has no
such submission left, it is dropped with a warning, so it never ends a unit
submitted after it arrived. If more than 256 events are held, the clock
drops all of them and holds nothing more for that epoch (logged as an
error): the epoch's later units are left to the timeouts until a cancel.
Progress that arrives while a later unit of its epoch is pending is
attributed to that unit, so a plugin's later stages must not emit progress
(e.g. a repeated segment end) for a unit after it is complete.
A hook that raises is logged once per session and counts as "no decision" /
"not complete", which leaves the unit to the timeouts. A settled speaking
unit keeps its place for its final-stage output, so attribution stays
correct after a stall. Each acknowledgement reports only units created up
to its input. A plugin whose input buffer can cut several units from one
append implements `PcmAppendBuffer.prepare_backlog`; the runner then submits
all of them for that append. The timeouts' defaults are the same for every
model (a per-plugin default is a possible follow-up). A plugin enables the mode with
`supports_input_clock = True` once those hooks match its real unit
boundaries; sessions of other models that ask for it are refused
(`input_clock_unsupported`). A plugin whose committed turn can split into
several Stage-0 submissions for some input modality (vision units it
replays separately, `prepare_commit_prefix`) names the modalities its
clocked sessions do accept in `input_clock_input_modalities`; the session
is then created without the others (`apply_input_clock_capabilities`), so
the modality check refuses them (`invalid_input_modality`). MiniCPM-o 4.5
opts in, for audio: its Code2Wav stage streams across unit boundaries, so
its `unit_output_complete` waits for the Talker segment end and the Code2Wav
output of the chunk that flushes the unit's remaining codes
(`tts_is_last_chunk`). The orchestrator reads each stage's stream
independently, so the flushes of several queued units can arrive before the
first Talker segment end while the engine hands every final-stage output to
the oldest speaking unit; the hook therefore counts the flags in `state` and
consumes them in unit order, the n-th flag completing the n-th speaking unit
(a Code2Wav segment end alone does not complete a unit, so a trailing
control-only turn-end snapshot of the previous unit cannot complete the next
one); a Stage-0 segment that
hands the Talker nothing (`<|speak|>` straight into `<|chunk_eos|>`, or the
chunk terminator drawn as the unit's first decision) is forwarded as an
empty hand-off, which the Talker closes like any segment, so every speaking
unit has a Talker segment end and a flagged Code2Wav output of its own and
the default `unit_decision` is enough; and its input buffer implements
`prepare_backlog`. The flag marks the end of what the bridge hands Code2Wav
for the unit, not of its audio: the vocoder's lookahead, and a remainder
shorter than 5 frames that the connector carries into the next unit's first
chunk, put the unit's last codec frames (usually 200–440 ms of audio; no
structural bound) into audio emitted with the next unit, as in real-time
operation, so the codec path is the same with and without the clock. Only
the chunk that ends the turn flushes everything; the API doc states this lag
in the model's entry.

What a plugin must guarantee before it opts in:

- Stage0 ends every accepted submission with exactly one segment end, in
  submission order, and the unit's decision (if any) is taken on that
  segment-ending output. An extra or missing segment end (e.g. a
  prefill-only append that produces none) shifts the epoch's later units onto
  the wrong segment ends: an acknowledgement may then arrive before its
  unit's output, or wait for the timeouts, until the next cancel starts a new
  epoch; the timeouts keep acknowledgements coming but do not resynchronise.
- The stage just before the final one (`final_stage_id - 1`) is the only
  one whose segment end sets a unit's `upstream_segment_finished`, and it
  ends exactly one segment per unit; an extra one is held for, or credited
  to, the next unit.
- With `final_stage_id == 0` (e.g. a text-only request), a unit completes at
  its Stage0 segment end whatever the hooks say.
- The default `unit_output_complete` assumes final-stage outputs of
  different units do not interleave (each unit's audio ends with its own
  segment end before the next unit's begins). A model whose final stage
  streams across unit boundaries must override it. Turns never overlap in
  an input-clocked session: the session is created without
  `supports_concurrent_turn_requests` (`apply_input_clock_capabilities`,
  the one place that decides it, so the advertised value and the runner's
  check cannot disagree), so a commit during a response waits for that
  response (the deferred-turn path),
  because the clock credits outputs to the speaking units in order.
  Crediting by request, which would let turns overlap, is future work.
- A unit is complete when everything the model emits for it has been
  emitted. A final stage that synthesises the end of a unit's output only
  together with later units' input (e.g. a vocoder lookahead, or a codec
  that emits several units per chunk) may complete the unit before that
  output, as real-time operation emits it; the lag is model-specific and
  each model documents its own in its model entry. The
  acknowledgement waits for a response continuation the output handler
  spawned (e.g. the `response.done` a non-resumable model sends from it);
  the wait is bounded (one second) and happens once per continuation, so a
  continuation that blocks is logged and acknowledged around rather than
  stalling the session, and delays only the acknowledgement that waited for it.

Every cancel that calls off appends of an input-clocked session advances the
epoch (an overlap barge-in does so too, where a session without the clock
keeps its epoch when nothing else was running), so a called-off submission
that already reached Stage0 has its output dropped instead of shifting the
count.

Frame-based plugins use `engine/duplex/intermediate.py::build_duplex_append_prompt`
for the shared request identity, sequencing and config snapshots. Token budgets,
PCM framing and model-specific worker fields remain in the plugins.

MiniCPM-o 4.5 preserves the complete reference waveform after resampling to
16 kHz. Its first-append context budget follows the whole-clip encoder:
ceil to 160-sample mel frames, ceil through the stride-2 CNN, then floor
through stride-5 pooling. Reference audio must not be cropped to a 100 ms
boundary to simplify budgeting: the remaining samples can change the
embeddings even when they do not add a pooled token.

The default `minicpmo_4_5.yaml` deployment selects MRv2 for all three stages
on CUDA, ROCm and NPU. The ROCm override changes runner selection while
preserving the shared duplex session, async-chunk, device, connector and stage
topology contracts. The CUDA-only reference-attention, FlashAttention, graph
and speculation behavior described below does not apply to ROCm. A ROCm FP16
Code2Wav configuration may select the shared fused CFM body with its SDPA
fallback; it does not enable NVIDIA tiled attention. Runner selection and this
dispatch coverage alone do not establish ROCm runtime or quality qualification.

MiniCPM-o native duplex does not apply a character cap unless
`max_speak_chars_per_chunk` is set to a positive value. The default is 0,
so the default length limit is the official 20-token budget; sampled chunk
terminators can still close a chunk earlier. A decoded prefix
ending in U+FFFD may continue to complete its UTF-8 bytes when a positive
cap is set. That cap keeps the token that crosses the target, then closes
the chunk before another draw. Hard token and request caps retain priority;
explicit `force_listen` bypasses only the soft character target.

The listen decision takes the released `streaming_generate` knobs. A session
created with `extra_body={"listen_prob_scale": s}` multiplies the listen logit
by `s` after the repetition penalty, and `extra_body={"listen_top_k": k}`
returns `<|listen|>` outright when it ranks within the top-`k` logits, before
the final draw. Both default to neutral (`1.0` / unset) and are validated at
session creation; they ride the append payload next to `force_listen` and
cannot change once the session exists. An early listen never enters the
repetition history, matching the reference decoder's early return.

MiniCPM-o preserves the stage-2 sampled `<|turn_eos|>`, including after text
without sentence-ending punctuation. The sampler does not infer whether an
answer is complete from its decoded tail or replace EOS with another candidate.
This matches the released `streaming_generate` turn-ending behavior and avoids
adding text to short answers such as `11` or `北京`. There is no worker-local
punctuation/continuation state for asynchronous lookahead to advance beyond the
scheduler's accepted span. The scheduler still supplies the canonical unit
content and terminator for the next append, including with sliding windows off.
Sentence-ending punctuation remains a Talker drain hint; it does not decide
whether the Thinker may end a turn.

An explicit `response.create` may open a latched assistant turn on its first
silence unit. The sampler consumes that opening marker once per request/unit,
even if the turn was already open, so repeated decode steps cannot undo a later
`<|turn_eos|>` decision. The consumed identity lives in the session state.
Opening a forced response permits the model to speak or listen. Its initial
listen decision is preserved and closes the response without audio; only an
already-speaking turn uses the normal listen-to-TTS continuation rewrite.

`<|turn_eos|>` does not terminate the input unit. The Thinker continues until
a chunk terminator, and the Talker receives all forwarded tokens and their
hidden states before that boundary, including any text after an intermediate
EOS. Such text or a later speak decision reopens the turn. The bridge and
Talker therefore report `turn_end` only when the final condition token is
`<|turn_eos|>`, not merely when EOS occurs somewhere in the unit. For example,
`<|speak|><|turn_eos|><|speak|>green<|chunk_eos|>` delivers `green` and keeps the
response open. Async lookahead after the actual chunk terminator is excluded.

If a speech unit has no text, its chunk boundary still
travels through Talker's empty-condition path. That path emits no audio but
preserves the TTS segment-end notification needed to schedule the next silence
unit; an empty speech chunk must not be dropped as if it were a listen unit.

Code2Wav attaches queued transcript text once per output payload. The data
plane deduplicates it by `(duplex_audio_cache_epoch, duplex_audio_chunk_seq)`
within a model turn, not by text equality or a shared prefix. Adjacent chunks
containing `1` and `1` must deliver `11`. Legacy outputs without a chunk
identity retain the cumulative-text snapshot handling.

The MiniCPM-o SigLIP vision tower defaults to reference attention for CUDA BF16 duplex
sessions. SDPA and eager produce different BF16 embeddings, which can change
scene descriptions during long video sessions. Turn-mode and non-CUDA vision
retain SDPA by default. An explicit engine or vision attention setting still
overrides the default.

The reference vision path fuses scaling and softmax for unmasked CUDA BF16
inputs, preserving BF16 score/probability rounding and FP32 accumulation.
Encoder CUDA graphs remain enabled. This avoids the full FP32 probability
buffer used by eager attention, but still materializes attention scores and
can reduce vision throughput compared with SDPA at larger image batches.
An explicit `vision_config._attn_implementation="sdpa"` keeps the SDPA path.

With reference vision attention, `vision_fused_layers: true` does not enable
fused encoder layers: `supports_fused_layers()` excludes this attention backend.
This also applies to the `minicpmo_4_5_duplex_h200.yaml` profile. Packed encoding,
vision CUDA graphs and the fused scaled softmax remain enabled in that profile.
Selecting SDPA explicitly allows fused encoder layers, but can change BF16 vision
outputs and requires rechecking long-video recognition for the workload.

MiniCPM-o deployments select `TRITON_ATTN` for CUDA stage 0, including the
standalone 2-GPU, 3-GPU and 8x4090 layouts and inherited replica and turn-mode
profiles. Talker and NPU attention settings are unchanged. This aligns the
Thinker backend selection; it does not establish accuracy or performance parity
across GPU types and layouts.

The inherited CUDA duplex profile uses upstream `vllm_c` RMSNorm and fused
add-RMSNorm, `silu_and_mul`, and QK-normalization/RoPE fusion for Thinker stage 0.
The checkpoint's model and head dtypes are retained. The turn profile resets
`kernel_config` and `compilation_config` to upstream defaults, including its
MRv2 derivatives. Other platforms and Talker keep their existing settings;
explicit engine overrides retain precedence.

Tool result conversion is preparation until the session's final deadline/epoch
check and admission. `runtime_config_for_function_output` receives an isolated
item copy with `name` and `arguments` from the immutable session call origin,
when available. Client-supplied hints for those fields are ignored; a plugin
requiring absent metadata rejects the result before admission. The client
acknowledgement retains the submitted item. Parsing and conversion should not
consume a separate call-metadata cache before admission.

A preparation refusal leaves the call pending and does not reset its deadline.
Missing origin metadata cannot be repaired with client hints: a plugin that
requires it keeps refusing retries until the call times out or is cancelled.
When the optional timeout is disabled, cancellation or session teardown is
required to retire such a call.

`on_tool_call_settled` handles metadata-only terminal cleanup, including old
epochs; `on_tool_calls_closed` disposes registered call metadata during teardown.
Model recovery uses the mailbox-ordered `runtime_config_for_abandoned_tool_call`
for a call that timed out or was cancelled in the current epoch, and for a repeat
the session refused to expose. The hook shows a fixed status, never client text.
It defaults to a no-op, and each model adapter decides which reasons it can
deliver. Qwen, Nemotron and JoyAI implement it. Showing a notice does not
establish how a model behaves afterwards; validate that per model. Accepted
results use their prepared candidate, while effective epoch cancellation uses
`retire_pending_function_output` for queued work according to model policy.

The ledger keeps a fingerprint (name plus canonical JSON arguments) of each
pending, timed-out or cancelled call. A pending call needs this too: Qwen keeps
tool pairs out of history until a result arrives, so the model sees an unanswered
question on the next turn and asks again.
`model_channel._expose_function_call` refuses a matching call before registering
it, so a suppressed repeat uses no call budget and arms no timer. A repeat of a
pending call gets no notice, because its result may still arrive. The ledger takes the session's `user_turn_seq` only when
`function_call_retry_policy` is `user_turn` and the model declares
`clears_function_call_block_on_user_commit`, because a commit is not always a user turn:
JoyAI commits are camera observations and native models free-run. An entry stays
blocking until that counter moves past the turn it was abandoned in.

`function_call_interrupt_policy: keep` gives each record a mutable `admission_epoch`;
`ToolCallOrigin` stays immutable. An epoch advance moves the epoch forward instead
of settling the call, and admission and timeouts read it.

JoyAI-VL-Interaction uses the same call book for a different decision shape. Its
model emits an action token (`</response> <note> </delegation> <question>`), so
`parse_function_call` reads a call the plugin itself projected, not a native tool
schema, and the declared function is one opt-in `delegate` entry. A recognised
call short-circuits the rest of that stage output, so the plugin projects the call
alone and the spoken note on its own ordinary path; the note therefore still
reaches the audio stage while the call is published. An accepted result never
enters `session.history`. It becomes a `[Delegated Results]` block in later JoyAI
prompts, and `has_pending_tool_followup` reports whether one is still waiting.
JoyAI's interaction is chunk based: a result is carried by observations until a
stored reply answers it, reused once inside the chunk that answered it, and
expired when the turn that asked it is cancelled. The chunk id and the
"a reply of it stored text" flag live in the session runtime config rather than on
the plugin, because one plugin instance serves every session of a deployment.
Every hook that the session installs returns the complete runtime, not just its
own keys: the session replaces the whole config, so a partial return would drop
the model's TTS and prompt settings.

Nemotron's function-channel wire output is request-aligned. In runtime batches,the worker-provided spans select each request's last scheduled hidden row rather
than CUDA-graph padding; the worker's payload split preserves that mapping for
the session data plane. Intermediate prefill chunks expose no function token
and keep empty slots in mixed batches. The final prefill chunk and decode frames
can expose function tokens. Model-local postprocess feedback alone does not
expose a client tool call. Forced feedback remains queued until an eligible
frame's postprocess consumes it once.

Turn plugins may bind their stage identities in `configure_pipeline` and select
an append's entry with `input_stage_id(payload)` (default 0). The request context
and orchestrator retain that entry stage. Session prompt preparation receives
the request's complete `stage_sampling_params`, so a pipeline with an ASR prefix
budgets the JoyAI prompt against JoyAI's generation settings. Session model config
comes from the first tokenizer-owning stage, matching frontend comprehension
ownership. JoyAI's audio-free requests enter its `joyvl` stage directly; speech
enters ASR and carries the original observation context through the handoff.

JoyAI renders a retained visual observation's unheard response as an assistant
message containing only `</response>`. The Session history item remains empty;
the model prompt keeps JoyAI's action marker to close the previous observation.
This preserves the turn boundary in both direct and ASR-handoff prompts without
restoring unheard text or treating cancellation as a model silence decision.
The boundary follows the existing recent-history budget and disappears when
its history item is deleted.

Both text and speech queries precede their observation's frames inside the same
user message, matching the visual reference frontend. The ASR handoff and
reconstructed history preserve this query-before-frame format.
Spoken observations retain the public Session item ID and its non-reusable
source ID. JoyAI resolves the current text only while both identities match.
Deleting and recreating a public item ID must not attach the new item's text to
old frames, even when the text is identical. Observation ordering anchors also
retain user source IDs so recreated items cannot inherit old positions.
The retained transcript precedes the original frames in the same user message,
matching the ASR handoff. A later
camera observation must not move the transcript into a separate user turn.
Deleted transcripts stay absent; cancelled output does not discard an accepted
user transcript.

The default JoyAI silence persona keeps ongoing user instructions active across
replies: a completed answer does not finish an instruction about future
observations. Each new observation must be evaluated against applicable ongoing
instructions before choosing silence. This does not force a spoken response for
every camera commit; silence remains valid when no instruction or scene change
calls for speech. The absence of a new query is not by itself a reason to stay
silent: an ongoing instruction may require a reply to the newest observation.
Answered one-time questions and unmet conditions in ongoing instructions do not
require another reply. An explicit `joyai_system_prompt` still replaces the persona.

The ASR handoff retains frame timestamps and observation-to-history references,
so an old frame precedes its reply boundary and the new transcript accompanies
only the current observation. Both paths apply the image limit before selecting
recent history. An empty boundary is omitted when none of its observation's
frames survives prompt assembly; pending consolidation records alone do not
keep it alive. Non-empty committed replies
still referenced by the observation ledger remain eligible at the recent-history
boundary even after their frames leave the prompt.
Frame eviction removes only the standalone generated timestamp paired with the
frame; neighboring user text remains, even when it ends with a timestamp.

For a speech-only follow-up without a new or working-window frame, the ASR
handoff includes the latest frame still awaiting visual consolidation. It keeps
that frame's original timestamp and observation-to-history references, before
the new transcript's turn. Once the frame has been summarized, the summary
provides its context instead of restoring the cached raw frame.

While idle, JoyAI retains the latest camera frame and one input chunk of audio
pre-roll. Once speech starts, it keeps the utterance until commit. An idle
camera commit enters JoyAI directly; reserved-byte accounting includes the
pre-roll until the commit is consumed or rolled back.

Completed input transcriptions travel with stage outputs through the runner
mailbox. After checking the request fence, the runner updates the exact committed
audio history item and emits its transcription event. The append captures the
original native audio part before yielding and retains that identity on its
request resource; a reused public item ID does not confer transcript ownership.
Cancellation, item deletion or replacement cannot apply a late transcript to a
newer turn.

See [supported models and deployments](../serving/full_duplex_api.md#enable-full-duplex)
for the current plugin integrations and deployment configurations.

## Heard-text history calibration

With `ack_only`, the client reports how much output audio it played. Some
models have no text-to-audio alignment for an interrupted reply, so the
existing history policy may retain an empty assistant turn. Optional
calibration estimates a prefix of that reply's original text before the next
prompt is built. It remains experimental and disabled by default; configuration
and strategy limitations are described in the
[serving guide](../serving/realtime_duplex_api.md#optional-history-calibration-for-interrupted-qwen3-omni-replies).

### Ownership and callback contract

`DuplexModelPlugin.history_calibrator(runtime_config)` returns either `None`
or a `HistoryCalibrationPolicy`. The plugin selects the strategy and declares
whether it needs retained audio through `requires_audio`. The runner creates
one `HistoryCalibration` controller per session on its own asyncio loop. The
controller owns recordings, task deadlines and result versions;
`DuplexEngineSession` owns the authoritative conversation and validates every
writeback. The callback must yield during slow work and must not mutate session
history.

The async callback receives an immutable `HeardTextSnapshot`: response ID,
original text, client-confirmed playback cutoff in milliseconds, and mono
float32 PCM from the reply's start with its sample rate. A text-only policy
receives empty PCM and a zero sample rate. `complete_audio_duration_ms` is
present only after the final audio stage finishes; an interrupted generation's
partial audio duration cannot estimate the rate of its complete text. The
callback returns an exclusive character offset into the original text (`0`
means an accepted empty prefix), or `None` to keep the existing policy. It
never supplies replacement history text or an ASR transcript to insert.

### Runtime sequence and guarded writeback

1. `ModelChannel` records projected output in the controller. Under `ack_only`,
   cancellation of a response or input, output-audio clearing, barge-in,
   truncation and a final playback acknowledgement can trigger `request_ready()`.
   Each prompt build also calls it through `before_prompt()`, re-evaluating
   eligibility on every append. Only ended replies with a positive, partially
   played cutoff are eligible; active replies and replies whose TTS still drains
   wait.
2. The controller captures the original text and cutoff, starts the callback
   with `asyncio.wait_for`, and identifies the task's evidence by
   `(text, cutoff, retained_audio_bytes)`. Changed evidence replaces the task;
   unchanged evidence is not retried after failure. Callback failure, refusal
   or timeout preserves the existing conservative history policy.
3. Before freezing the stage request's runtime and preparing its conversation,
   `ModelChannel._append_via_data_plane` awaits `before_prompt()`. This has one
   `history_calibration_timeout_ms` budget (default 1000 ms) for all eligible
   tasks, including replacements; replacement work does not extend it. At the
   deadline, the controller revokes task ownership and cancels pending work.
   The append then checks the session epoch and open state before continuing,
   so cancellation or close during the wait cannot submit a stale prompt.
   History and tool runtime are captured after the wait, so a calibration that
   consumes a tool pair cannot leave that pair beside its stored answer in
   the submitted prompt.
4. A result may write back only while its controller is open, its task still
   owns that response and its evidence version still matches. The session
   additionally checks that the original response snapshot and conversation
   item exist, the original text is unchanged, the offset is in range, and
   the result's audio cutoff does not exceed the current permanent cutoff.
   Accepted results replace text at the original assistant item's position in
   history.
   Deletion, replacement, stricter truncation and close therefore reject stale
   results even if cancellation of an async callback arrives late.
5. After an accepted writeback, the controller notifies the runner. If the
   stored assistant item has nonempty text, the runner settles that response's
   runtime as committed, just as a playback ACK does. Qwen then consumes its
   tool pair without requiring another ACK. An empty prefix keeps the pair
   available for a retry, including late ACKs after its evidence is retired.
   Settlement runs outside the calibration-failure handler. A plugin failure
   while settling a stored answer emits `tool_call_settlement_failed` and closes
   that session; it cannot continue with history and tool runtime out of sync.

### Evidence lifetime and limits

The controller retains at most four replies and bounds retained PCM by
`history_audio_max_bytes_per_session` (default 8 MiB). It drops whole replies
rather than retaining a suffix that could be mistaken for audio starting at
zero. Text-only calibration retains no PCM. The byte limit excludes temporary
callback copies and does not bound the conversation's stored text.

Eviction and item deletion discard recordings and revoke pending tasks. For an
ended reply, `retire_history_calibration` also releases its original response
snapshot, playback cursor and pending reconciliation state. A still-generating
or draining reply defers retirement until its output no longer needs that
state. The stored prefix and compact cutoff metadata remain with an undeleted
conversation item. Late ACKs cannot expand a retired prefix; a later, stricter
explicit truncation clears it conservatively because its alignment evidence
has expired. Finishing calibration alone does not retire evidence that a
stricter truncation may still need. Session close closes the controller,
cancels its tasks and releases its recordings.

## Serving

`OmniDuplexSessionHandler.handle_realtime_session(websocket)`:

1. accept; build the `RealtimeEnvelope` (autostart / default session from
   query params, `event_id` correlation, error rendering through the typed
   `ErrorEvent`);
2. handshake: first message `session.update` -> `omni.open_session(session)`
   (the id inside the payload is ignored; the allocated id is announced in
   `session.created` together with `resume_token` / `attachment_generation`
   when the model supports resume); `session.resume(session_id, resume_token,
   last_received_server_event_seq)` -> `attachment.authenticate_resume` ->
   `omni.resume_session`;
3. reader loop: JSON -> `RealtimeEnvelope.translate` (`translate_realtime_command`
   with the session's declared audio defaults) -> `handle.submit`;
   envelope-level errors (invalid JSON, oversize frame, unknown type,
   event acks) are answered locally;
4. writer pump (session-scoped, survives reconnects): `async for ev in
   handle.events(): attachment.send_event(ev.to_realtime())`, journaling
   for replay until the journal overflows (`session.resync_required`);
5. disconnect: a resumable session is detached (`attachment.detach` +
   `omni.detach_session`, engine-owned grace); a superseded socket's
   disconnect is a no-op; a non-resumable session is closed. Takeover sends
   `session.replaced` to the old socket and closes it.

## Validation

- CPU suites: `tests/engine/duplex/**` (session, lease, manager, runner,
  plugin, events, commands), `tests/engine/test_duplex_orchestrator.py`,
  `tests/engine/test_duplex_omni_engine.py`,
  `tests/entrypoints/test_duplex_omni.py`,
  `tests/entrypoints/duplex/test_duplex_serving.py`,
  `tests/entrypoints/openai/test_duplex_session_attachment.py`,
  `tests/entrypoints/openai_api/test_duplex_api_server.py` (the duplex
  server: pipeline probe, app state, routes, warmup gate),
  `tests/entrypoints/duplex/test_duplex_warmup.py`,
  `tests/entrypoints/duplex/test_startup_warmup.py`,
  `tests/clients/**`, `tests/engine/test_duplex_import_boundary.py`,
  `tests/model_executor/models/minicpmo_4_5/duplex/**`,
  `tests/worker/test_native_duplex_hooks.py`.
- Calibration CPU suites: `tests/engine/duplex/test_history_calibration.py`
  covers cache eviction, live-suffix rejection and deferred retirement.
  `tests/model_executor/models/qwen3_omni/test_history_integration.py` covers
  task ownership, cancellation, prompt deadlines, failed-snapshot deduplication
  and the calibrated history in the next submitted prompt. The remaining Qwen
  `test_history_*.py` files cover prefix matching and token-rate estimates.
- GPU: `examples/online_serving/minicpmo/realtime_duplex_demo.py` (one turn
  over the websocket), `examples/online_serving/barge_in_client.py` (cancel
  during playback; `--inline` drives the same scenario through
  `InlineDuplexClient`), `tests/e2e/online_serving/run_minicpmo_realtime_duplex_multi_session.py`
  (admission, resume, takeover, expiry).

## Realtime-Venus delegation speech condition

Realtime-Venus selects the native duplex engine by default through
`realtime_venus_omni.yaml`;
`realtime_venus_omni_turn.yaml` selects the ordinary turn engine. Both
profiles inherit the corresponding MiniCPM-o duplex or turn deployment,
including its Flow precision, step count and stage settings.

Realtime-Venus reuses the MiniCPM-o stage implementation. Its optional
`delegate_open_token_id` and `delegate_close_token_id` belong to the model
plugin's special-token metadata. At the native Thinker-to-Talker handoff,
the bridge removes delegation delimiters and their enclosed token/hidden-state
rows from speech conditioning while preserving the complete text output.
Delegation parsing uses the full current unit, including a delimiter sampled
in the opening decision slot. That decision updates the mute state even when
it has no speech-condition row; the remaining speech rows retain MiniCPM's
original alignment. Delimiter-only units still carry their text and mask
transition, so a closing decision in a later unit can resume speech.
The mask is request-local, continues across units, and resets at a new turn
or the turn-end token. Delegation can also mask the turn-end speech row;
dialogue termination is derived from the original Thinker unit.
Models without these optional IDs retain their existing handoff.

The Venus session plugin additionally projects one completed delegate span per
model turn as the shared Realtime `delegate(question)` function. Parsing is
request/turn-local and accepts spans split across text deltas. A repetition of
the same completed span within that model turn cannot execute a second call.
The application
executes the call; the shared session ledger owns result admission, deadlines,
repeat suppression and interruption policy. MiniCPM-o continues to advertise
`tool_mode: none`.

Accepted Venus results are tokenized as `<backend>result</backend>` into a
bounded private runtime queue. The delimiters are encoded separately; reserved
token spellings inside the result body are encoded as ordinary text, preserving
their readable content without adding model control tokens.
Results wait for a projected turn-end or genuine model-listen boundary;
synthetic segment-complete listen events only schedule another unit and leave
the result queued. Closing a delegate span alone does not end the foreground turn.
An identical completed span repeated in a later unit also permits injection,
so a model waiting for its backend cannot strand the accepted result in a loop.
The next append reserves their exact token count and feeds them after that
unit's media in the existing Thinker KV. They also
belong to the unit's sliding-window history and reopen the speech gate after a
completed turn. Ownership transfers to the stage only on successful submission;
the commit removes only the submitted generations, retaining newer arrivals.
Cancellation clears unsubmitted results; submitted requests retain the ordinary
epoch fence. This is native duplex continuation, not HTTP `tool_calls` or the
external Venus Harness/memory runtime.

A fully muted unit following speech sets model-owned `meta.speech_flush`.
The Talker appends only its audio-begin embedding to the existing condition,
with a 26-token maximum and no minimum. The codec bridge drains pending frames
and closes the vocoder stream, but does not set the dialogue `turn_end` or mark
that dialogue turn as already terminal. These flags travel with the condition's
sequence identity so delayed outputs cannot flush a later condition. The flag
is explicitly false on subsequent units; it cannot stick through metadata merge.

Repeated muted units and a bare turn terminator after the flush produce no new
codec tokens. Speech resuming in the same dialogue turn replaces the Talker
prompt/KV and starts a fresh vocoder stream. The first speech after initial
delegation-only units also starts with fresh Talker prompt/KV and a zero minimum
codec-token budget, even though there was no earlier speech to flush. Subsequent
speech units continue normally. A new turn or request resets all delegation
state. This keeps speech lifecycle independent of dialogue lifecycle.

`meta.text_without_audio` opts Venus units into text delivery independent of
speech generation. It travels through Talker and Code2Wav for every unit,
including mixed delegation/speech units that sample codec EOS before producing
any audio frames. The codec bridge consumes their pending text at the unit
boundary; the data plane emits it once using the same cache-epoch/chunk-sequence
deduplication as audio. It does not wait for later speech or replay the text with
the next audible unit. Stock MiniCPM keeps its existing audio-gated delivery.
Zero-frame flushes on a live native request carry a scheduling placeholder so
the generation receiver retains their text and boundary metadata, including
speech-only flushes with `last_chunk=True`. `code_flat_numel=0` keeps the
placeholder out of vocoder input; it does not close the resumable request.

This filtering does not execute delegated work or supply the external Venus
memory runtime. The [Realtime-Venus recipe](gh-file:recipes/inclusionAI/Realtime-Venus.md)
defines the currently exposed serving scope and validation limitations.
