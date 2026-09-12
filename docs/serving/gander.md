# Gander full-duplex dialogue

[Gander](https://github.com/Omni-Interaction-Gander/Omni-Interaction-Agent)
uses the MiniCPM-o 4.5 Thinker, Talker and Code2Wav pipeline. This integration
uses the repository's vLLM 0.29 environment. It provides model inference and context management, not
external tool execution, trusted ASR binding or a Brain/Gateway service.

## Prepare and serve

The release contains separate Thinker and Talker components. Compose a model
directory without modifying the downloaded weights:

```bash
hf download Gander-Omni/Gander --revision 24fc4cc8543f95daf99be53b6199403a7732688f
python -m vllm_omni.model_executor.models.minicpmo_4_5.gander \
  /path/to/downloaded/snapshot /path/to/new/gander-model
vllm serve /path/to/new/gander-model --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/gander.yaml --port 8091
```

Keep the source snapshot available: the composed directory links to its weights
and reference voice in `assets/ref_audio.wav`. Connect through
`WS /v1/realtime?duplex=1` or `WS /v1/duplex`; see
[Realtime API](realtime_api.md) for session setup and audio events.

Input is mono PCM16 at 16 kHz. Model input units are one second; upload packets
may be smaller. Continue microphone input, including silence, while awaiting
the model's response. Input-unit length does not determine output audio length.
The model chooses listen, speak, backchannel or interrupt. A model interrupt
cancels the active response; clients must honor playback-clear events.

## Tools and context

Tools are optional function schemas in `session.update.session.tools`.
The model emits Realtime function-call events; the application executes calls
and returns results using `conversation.item.create` with a
`function_call_output` item. No tool is executed by KV replay.

Custom context events use stable `event_id` and the current `epoch`:

- `input.context.append`: a `runtime_event` or `tool_result` observation.
- `input.context.append` with `kind: task_slate`: replace the protected slate.
- `input.context.get`: inspect unit IDs, epoch, context version and resource generation.
- `input.context.replace` with `kind: history_edit`: apply versioned
  `pin`, `unpin`, `move`, `delete` or external-event `insert` edits.

For example, delete an unpinned unit using IDs from the current snapshot:

```json
{"type":"input.context.replace","context":{"kind":"history_edit","event_id":"edit-1","epoch":1,"base_version":2,"edits":[{"op":"delete","unit_id":"u0-3"}]}}
```

The Realtime endpoint wraps custom output events with `duplex.` and places
their payload in `event`. `input.context.appended` acknowledges queueing;
`input.context.applied` reports prefill progress. `input.context.replaced`
confirms reconstruction and supplies the new epoch for subsequent commands.

Replacement rebuilds scheduler-owned KV from selected inputs and completed
assistant tokens; it does not splice raw KV tensors or replay historical audio.
The system, tools, reference voice and latest slate remain protected.
Identical committed replacement IDs are deduplicated; conflicting IDs fail.
Validation failure preserves old KV. Failure after replacement begins closes
the session and requires reopening. Older reconnect cursors require
resynchronization.

Default history rollover starts at 128 units and retains 96; configure
`extra_body.gander_history.max_units/retain_units` for a smaller window.
Automatic rollover waits for active-response delivery. Hard context and replay
byte/token limits still apply; the deploy configuration allows a 256 MiB replay
journal per session. The configured admission limit is four sessions, not a
hardware-independent capacity or latency guarantee.
