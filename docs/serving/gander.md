# Gander full-duplex dialogue

[Gander](https://github.com/Omni-Interaction-Gander/Omni-Interaction-Agent)
finetunes MiniCPM-o 4.5. Its Unit8/50 dialogue profile uses the existing
scheduler-owned MiniCPM Thinker, Talker, and Code2Wav pipeline, with a
Gander-specific dialogue grammar and speech budget.

This is an experimental dialogue integration, pinned to the validated
transactional-duplex dependency from [PR #7294](https://github.com/vllm-project/vllm-omni/pull/7294)
at `36142ba106cdf458dfc77439cc8f769a68d1892d`. Later revisions of that PR are
not included. H200 validation covers streaming speech, native interruption,
tool/result exchanges, and physical KV reconstruction. Audio parity with the
complete official runtime has not been established. See the
[validation record](../design/gander_validation.md) for results and limits.

## Audio units and emission

The input model unit is 1000 ms (16,000 samples at 16 kHz). Client upload packets
are independently configurable; the example sends 200 ms packets. Gander's
non-final Talker budget is 50 S3 codes, whereas Code2Wav emits nominal chunks of
25 codes. The released 25 Hz tokenizer / 50 Hz mel / 24 kHz waveform path maps
25 codes to about 1 second and 50 codes to about 2 seconds of output audio.
An input unit is therefore not the duration of its generated speech.

The steady PCM chunks observed in vLLM-Omni are 1 second. Its first chunk can be
0.84 seconds because 160 ms is retained for vocoder overlap; terminal chunks and
WebSocket packets may have other sizes. The upstream minicpmo-utils 1.0.6
Token2wav instead prepends 160 ms of silence to compensate the first non-final chunk.
The release model card's "25 tokens, approximately 0.5 seconds" statement does
not match this pinned codec implementation and the observed PCM sample counts.

For the mixed long/short soft-interrupt fixture, require non-empty audio for
each response and actual multi-packet output from at least one response, not
at least two packets for every short answer. The new Gander-specific regression
keeps response-lifecycle checks separate from model-native cancellation. The
final H200 run observed an actual model interrupt, cancelled the old reply,
and delivered the follow-up answer with one audio packet and no stale audio.
This is a functional contract, not an interruption-latency guarantee.

The first action in each one-second input unit is `listen`, `speak`,
`backchannel`, or `interrupt`. Speaking units contain at most eight lexical
tokens and produce 50 S3 tokens (24 kHz output). The final speaking
unit drains to codec EOS within the remaining Talker context. Audio input is mono PCM16 at 16 kHz.

## Prepare the release

The repository root on Hugging Face is not a Transformers model directory.
It contains separate materialized `thinker/` and `talker/` components.
Download the matched release and compose a symlink-backed directory; this
neither merges tensors in memory nor changes the downloaded snapshot.

```bash
# Set HF_ENDPOINT=https://hf-mirror.com if needed.
hf download Gander-Omni/Gander --revision 24fc4cc8543f95daf99be53b6199403a7732688f
python -m vllm_omni.model_executor.models.minicpmo_4_5.gander \
  /path/to/downloaded/snapshot /path/to/new/gander-model
```

Keep the source snapshot available: the composed directory links to its
weights and assets. The composer enables the `gander_unit8` config flag;
base MiniCPM-o deployments retain their existing behavior.

## Serve and test

```bash
vllm serve /path/to/new/gander-model --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/gander.yaml --port 8091

python examples/online_serving/minicpmo/realtime_duplex_demo.py \
  --url 'ws://localhost:8091/v1/realtime?duplex=1' \
  --model /path/to/new/gander-model \
  --input-wav question_with_trailing_silence_16k.wav \
  --ref-audio /path/to/new/gander-model/assets/ref_audio.wav \
  --output-dir /tmp/gander-dialogue --require-audio
```

Append microphone silence after the question to allow the model to finish
its response while continuing to consume one-second input units. Inspect
`session.created.session.capabilities` to confirm the duplex endpoint.
The same runtime is available through `WS /v1/duplex`.

```bash
export GANDER_MODEL=/path/to/new/gander-model
pytest -sv tests/e2e/online_serving/test_gander.py \
  -m 'advanced_model and cuda' --run-level advanced_model
pytest -q tests/model_executor/models/minicpmo_4_5/duplex/test_gander.py
```

The default deployment admits four sessions on a large-memory GPU. Tools are
opt-in. Brain, Gateway execution, and trusted ASR binding remain external.
The Gander model policy owns bounded history and protected-prefix replacement.

## Tools and runtime observations

Advertise standard function schemas in `session.update.session.tools` (or
`extra_body.realtime_tools`). Without custom instructions, the adapter uses
Gander's interaction/tool prompt with one call per unit. The model generates
calls; an external application executes them. Function output is validated
and emitted through standard Realtime function-call events, bypassing TTS.
`task_start` names a task; the official runtime additionally binds a trusted
user turn, which this inference adapter does not provide.

Return a tool result through the standard protocol:

```json
{"type":"conversation.item.create","item":{"type":"function_call_output","call_id":"call_...","output":"result data"}}
```

Append progress in the current epoch:

```json
{"type":"input.context.append","context":{"kind":"runtime_event","event_id":"progress-1","epoch":0,"call_id":"call_...","output":{"status":"running"}}}
```

An identical custom event ID is idempotent. `input.context.appended` acknowledges
queueing; `input.context.applied` reports the cumulative version already
prefilled. The Realtime endpoint wraps custom events with `duplex.` and puts
their payload in `event`. A complete audio unit must initialize the session first.

## Protected slate replacement

Set optional `extra_body.gander_task_slate` at session creation. Later updates
replace the prefix and rebuild KV, rather than appending another slate string:

```json
{"type":"input.context.append","context":{"kind":"task_slate","event_id":"slate-1","epoch":0,"version":1,"slate":"lookup: completed. No running tasks."}}
```

The operation emits `input.context.replaced` with a **new epoch**, applied
`context_version`, physical `request_id`, `retained_unit_ids`, `dropped_unit_ids`,
and `replayed_tokens`. Subsequent custom events must use the new epoch.
The bundled `DuplexClient` updates its session epoch from this receipt.
The underlying external task is not cancelled by this frontend context change.

If speech is active, replacement cancels it and emits
`output_audio_buffer.cleared`; clients must stop buffered playback. No old
response audio/text may follow the replacement fence. Set `generate:true` to
request a new model decision after replacement, without a fake audio chunk.

## History editing and rollover

Inspect stable unit identities:

```json
{"type":"input.context.get"}
```

The `input.context.snapshot` response includes the current epoch/version,
physical resource generation, context-token count, and unit summaries. Raw
reference audio, prompts and hidden state are not returned.

Apply edits atomically against that version:

```json
{"type":"input.context.replace","context":{"kind":"history_edit","event_id":"edit-1","epoch":1,"base_version":2,"edits":[{"op":"pin","unit_id":"u0-1"},{"op":"move","unit_id":"u0-7","before":"u0-2"},{"op":"delete","unit_id":"u0-3"}]}}
```

`before:null` moves/inserts at the end. Pinned units must be explicitly unpinned
before deletion. The system/tools/reference/slate prefix cannot be deleted.
Insert a registered call's validated external observation at a historical unit:

```json
{"type":"input.context.replace","context":{"kind":"history_edit","event_id":"insert-1","epoch":2,"base_version":3,"edits":[{"op":"insert","unit_id":"progress-history","before":"u0-7","event":{"kind":"runtime_event","call_id":"call_...","output":{"status":"running"}}}]}}
```

Tool results and progress can also use `input.context.replace` directly, or
`preempt:true` on `input.context.append`, to interrupt current generation and
rebuild before taking the next model decision.

The default Gander policy triggers rollover at 128 units and retains 96 recent
or pinned units. Automatic rollover waits for an active reply to finish
delivery; the unit threshold may briefly be exceeded, but hard token/byte
limits still apply. A smaller test/deployment policy can be configured at creation:

```json
{"extra_body":{"gander_history":{"max_units":8,"retain_units":4}}}
```

Token/byte budgets can force earlier eviction of unprotected units. Replay
re-encodes retained multimodal inputs and teacher-forces completed model output;
it does not replay tool execution or historical audio to the client. KV is
reconstructed through normal scheduler-owned prefill with new positions.

A validation error preserves old KV. A failure after physical replacement
begins closes the session safely; reopen to recover. Reconnect cursors older
than the replacement boundary require resynchronization, so stale audio cannot
return through journal replay.

See [the architecture, guarantees, and limits](../design/gander_agent_runtime.md).

## Validation contracts

The Gander interaction prompt is used for dialogue with or without tools.
The native-interrupt E2E requires an actual model interrupt event, cancellation
of the old response, no old audio after its terminal event, and the correct
follow-up answer. MiniCPM's original non-cancelling soft-interrupt contract
remains unchanged for its callers.

`test_gander_default_history_window` streams 140 seconds after a historical
edit to exercise the default window, including invalid-edit rejection and
inference after reconstruction. Snapshots wait for preceding input submissions.
The Gander deployment bounds replay metadata to 256 MiB per session; this
includes reference and tool-prefix snapshots. Byte-pressure compaction makes
headroom instead of repeatedly rebuilding after every new input.

## Browser dialogue and local tool demo

The optional [Gander Live app](https://github.com/vllm-project/vllm-omni/blob/main/apps/gander_live/README.md) runs separately
from the inference engine. It supports microphone/camera input, response-specific
playback cancellation, visible function-call events, and a bounded local lookup
tool. It can proxy a remote GPU service through an SSH tunnel.
