# AURA_v2 on the Unified Full-Duplex Framework

## Status and target

AURA_v2 is a proactive audio-visual assistant: it watches a live video stream,
decides on every turn whether to speak, and answers in speech. It is served by
the [Unified Full-Duplex Framework](fullduplex.md) through
`AuraDuplexPlugin` (`vllm_omni/model_executor/models/aura_omni/duplex/plugin.py`),
the same plugin as AURA v1. The thinker family (v1 or v2) is chosen by the
Stage1 stop ids in the deploy config, not by a separate pipeline.

```text
/v1/realtime?duplex=1
  -> DuplexSessionRunner (one per session)
       plugin.plan_append: one client commit -> one ephemeral four-stage request
  -> Stage0 Qwen3-ASR              speech -> transcript (vision-only turns get a silent pad)
  -> Stage1 AURA_v2 thinker        transcript + frames + session history -> <|silent|> or <|respond|> + text
       two-token logits processor (silent/respond gate, then the reply)
  -> Stage2 Qwen3-TTS Talker       whole reply text -> codec frames (skipped on <|silent|>)
  -> Stage3 Qwen3-TTS Code2Wav     codec frames -> 24 kHz PCM
  -> response.output_audio.delta + response.output_audio_transcript.delta
```

Deploy profiles: `vllm_omni/deploy/aura_omni_v2.yaml` (one GPU, eager, one
session) and `vllm_omni/deploy/aura_omni_v2_2gpu.yaml` (Stage1 on its own GPU,
CUDA graphs, up to 16 sessions). All four stages select Model Runner V2
(`AuraTwoTokenLogitsProcessor` on Stage1). Stage0 and Stage1 set
`async_chunk: false` so the orchestrator-fed ASR→AURA path does not park in
`WAITING_FOR_CHUNK`; Stage2→3 keep deploy-level `async_chunk: true` for codec
handoff.

## Supported scope

- Push-to-talk speech plus video frames, and vision-only commits (frames with
  no speech) for proactive replies; video is required, audio is optional.
- One ephemeral request per commit (`supports_turn_commit_only`). The next
  commit may start while the previous reply's Talker and Code2Wav still drain
  (`supports_concurrent_turn_requests`).
- Function calls parsed from the AURA_v2 tool-call format, executed by the
  client, and answered with a spoken follow-up.
- Multi-turn context kept per session on Stage1.

It does not claim:

- typed text turns (the only text-only turn is the tool follow-up);
- server-side VAD (the client commits).

Stage1 prefix caching is on. The thinker sets `uses_native_kv_cache_only`,
and the deploy profiles set `mamba_cache_mode: align`, so a hit restores both
the attention KV and the GDN state through vLLM's own cache. The omni
single-group output cache stays off: hybrid attention has more than one KV
group, and Stage1 only needs the text.

## Turn decision: two-token logits

AURA_v2 starts every reply with one control token. `AuraTwoTokenLogitsProcessor`
(`two_token_logits.py`) is registered on Stage1 through `logits_processors` and
opted in per request with `extra_args={"aura_mode": "streaming"}`, which the
plugin sets for v2.

| Decode step | Allowed tokens | Result |
| --- | --- | --- |
| First | silent (248070) and respond (248077) only | silent is a stop id, so the turn ends silent |
| After respond | everything except the two control ids | the reply body |

On a tie the lower id (`<|silent|>`) wins. The Model Runner V2 processor
decides the first step from `total_len == prompt_len` on device and does not
read the output list. The shipped profiles keep Stage1 `async_scheduling:
false`; whether V2 still needs that setting is not measured. Stage1 keeps the
temperature from the deploy profile (the shipped profile uses 0.2). The plugin
forces ``min_tokens`` to 0 and the penalty fields to their no-penalty values so
a penalty cannot push the control token; it does not override temperature.

## Control-token families

v1 and v2 tokenizers share ids: v2's vocabulary still contains v1's silent id
`151669` as an ordinary piece. Stopping on both families would truncate v2
speech, so the two are never merged (`control_tokens.py`).

| Family | Thinker | silent | im_end | respond |
| --- | --- | --- | --- | --- |
| v1 | `AuraQwen3VLForConditionalGeneration` | 151669 | 151645 | five ordinary pieces |
| v2 | `AuraQwen3_5ForConditionalGeneration` | 248070 | 248046 | 248077 (not a stop) |

`profile_for_stop_ids` picks the family from the Stage1 `stop_token_ids`; an
empty list keeps v1. `stop_ids_for_profile` then drops the other family's
silent / im_end pair and `<|respond|>` from the stop list. The family also
selects the v2 system prompt and the v2 generation prefix, which ends with the
closed `<think>\n\n</think>\n\n` block of the checkpoint's chat template
(`enable_thinking=False`).

Stage1 runs with `skip_special_tokens: false` and
`include_stop_str_in_output: true` so that `<|silent|>` stays visible to
`decide_output`. Before text reaches the Talker, `_strip_assistant_text`
removes the control tokens and the grounding label tags
(`<|object_ref_start|>` / `<|object_ref_end|>`), keeping the object name. The
box span (`<|box_start|>…<|box_end|>`) stays in the client text for drawing and
is not spoken. Session history keeps the raw text.

## Stage1 to Talker

The orchestrator feeds the Talker from Stage1 through `aura2tts`, not a
`from_stage_1` connector edge: that edge parks the Talker in
`WAITING_FOR_CHUNK` and clears its text conditioning. Only Stage2 to Stage3
uses a shared-memory connector (`async_chunk: true`).

`sentence_tts.plan_partial_stage_output` hands the reply to the Talker in one
submit when Stage1 finishes. Qwen3-TTS CustomVoice bakes the text present at
the first Talker prefill, so a clause sent earlier would be the only thing the
user hears. Code2Wav is prewarmed on the same request id. Silent finishes,
unclosed `<think>` and tool-call blocks emit nothing.

When Stage1 finishes, `release_concurrent_turn_requests` opens the input gate:
the next commit starts a new request and a new `response_id` while the prior
reply's Stage2 and Stage3 keep running (`draining_stage_ids` = {2, 3}).

## Function calls

The AURA_v2 chat template wraps calls in `<tool_call>` with a nested-tag body
(`<function=name><parameter=p>value</parameter></function>`), not the JSON body
that vLLM's Hermes parser expects, so `duplex/tools_utils.py` has its own
parser. Realtime `session.tools` (flat `{type, name, description, parameters}`)
are mapped onto the chat-template shape `{type, function: {...}}`.

```text
Stage1 emits <tool_call>             -> response.function_call_arguments.done
                                        (turn released when nothing is spoken before the call)
client runs the tool                 -> conversation.item.create (function_call_output)
plugin stores it as aura_tool_pending
client sends response.create         -> _plan_tool_followup: tool result appended to history,
                                        Stage0 gets a silent pad, Stage1 answers, Talker speaks
```

A call with spoken prose before it keeps its TTS and ends on the normal path.
`function_call_releases_turn` lets the shared runner free the request when the
call is the whole turn, so the follow-up `response.create` is not rejected as
`response_already_active`.

## Session history

`duplex/history.py` keeps one ChatML history per session in the Stage1
process. Each finished turn commits the assistant text (silent turns commit
`<|silent|>`); a user turn that carried frames keeps them and renders one video
placeholder. Tool results are appended as `<tool_response>` user turns before
the follow-up. When the history passes 45 video turns, the oldest 30 are
dropped, matching the reference gateway's trim.

## Startup warmup

The shared duplex warmup sends a short tone plus one frame. AURA_v2 hears an
empty transcript and answers `<|silent|>`, so the Talker and Code2Wav never
run before the first user. `startup_warmup_turn` replaces the tone with a
bundled 2.5 s spoken request (`duplex/warmup_request.pcm`, 16 kHz PCM16,
shipped as package data) so the warmup turn reaches all four stages. Without
the file the shared tone is used.

## Code locations

| Concern | Path |
| --- | --- |
| Thinker wrapper | `vllm_omni/model_executor/models/aura_omni/qwen3_5.py` |
| Control tokens | `vllm_omni/model_executor/models/aura_omni/control_tokens.py` |
| Two-token logits | `vllm_omni/model_executor/models/aura_omni/two_token_logits.py` |
| Plugin, capabilities | `vllm_omni/model_executor/models/aura_omni/duplex/{plugin,capabilities}.py` |
| Stage1 to Talker | `vllm_omni/model_executor/models/aura_omni/duplex/sentence_tts.py` |
| Tool calls | `vllm_omni/model_executor/models/aura_omni/duplex/tools_utils.py` |
| Stage processors | `vllm_omni/model_executor/stage_input_processors/aura_omni.py` |
| Deploy profiles | `vllm_omni/deploy/aura_omni_v2.yaml`, `vllm_omni/deploy/aura_omni_v2_2gpu.yaml` |
| Recipe | `recipes/AURA/AURA_v2-H200.md` |
