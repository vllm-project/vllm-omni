# Qwen3-Omni Input Transcription

Use the already-loaded Qwen3-Omni Thinker to transcribe committed user audio.
Open WebUI Realtime needs the standard WebSocket transcription events on
turn-mode `/v1/realtime`. The same recognition path also serves
`POST /v1/audio/transcriptions` and an optional Chat Completions SSE extension.

This page is for an operator installing this fork into an existing pipx runtime,
setting the served alias and client options, and judging capability, latency,
and remaining verification. It does **not** claim a live English/German or Open
WebUI persistence run.

!!! warning "Operator-reported working, not fully verified"

    An operator reports that text chat and Realtime voice work on the
    `0.32.0-owui` stable branch after an integer sample-rate resampler
    fix. Automated tests, new-logging syslog checks, and detailed
    language or Open WebUI persistence coverage have not been run here.
    Treat the contracts below as implemented source plus that operator
    report, not a completed verification suite.

## Prerequisites

Install this checkout into the existing pipx-managed Zenith environment. OmniKit
and installer changes are out of scope. Matching vLLM is pinned in the
[CUDA GPU install notes](../getting_started/installation/gpu.md): vLLM 0.32.0 at
commit `5df63a7203cdb9328d824a27b682e77c7b9928be`. Upstream requires
`openai>=2.25.0`.

Required serving shape:

- Existing Qwen3-Omni pipeline with `session_mode: turn` (not duplex).
- Served model name `zenith` for
  `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound`. The client transcription
  `model` field must be that served alias, not a second ASR model id.
- No extra ASR process, weights reload, HTTP loopback, or per-turn disk write.
- If the session advertises function tools, start the server with
  `--enable-auto-tool-choice` and a configured `--tool-call-parser`. The
  Realtime handler rejects function tools when that parser is missing. For
  Qwen3-Omni, use `--tool-call-parser hermes`: the duplex tool path uses
  vLLM's Hermes parser and `<tool_call>` markup. An operator reports that
  those launch flags cleared a separate tools problem on this deployment.
  Automated tool-execution tests are still unrun, and delegation tool
  execution is not claimed.

Base of this fork: `5917730a4b02aedb44d374173cdc1b10484ccd52` on branch
`0.32.0-owui`. Closed unmerged PR #3526 (commits `095bb79` / `77470c3` by Nick
Pratley) is starting material for wrapper delegation and HTTP/Chat inline
recognition, not a complete Realtime integration and not a publication target.

## Open WebUI Realtime

Open WebUI v0.12.0 waits for a nonempty
`conversation.item.input_audio_transcription.completed` event, then internally
calls `bridge.respond`, which becomes a standard provider `response.create` with
`metadata.input_item_id`. `bridge.respond` is not a provider event to implement.
HTTP transcription by itself does not start that path.

Configure the Realtime session:

```json
{
  "type": "session.update",
  "session": {
    "type": "realtime",
    "audio": {
      "input": {
        "format": { "type": "audio/pcm", "rate": 24000 },
        "transcription": {
          "model": "zenith",
          "language": "de",
          "prompt": "optional hint"
        }
      }
    }
  }
}
```

- A new session has no transcription settings, so recognition is off.
- Send `audio.input.transcription: null` to disable it.
- A later partial `session.update` that omits `transcription` keeps the
  current enabled settings.
- `model` must be a served alias such as `zenith`.
- `language` and `prompt` are optional. `language` must be a Thinker-supported
  code; unknown values are rejected. `de` and `en` are operator priorities,
  not an observed Open WebUI setting and not the full language map.
- Wire format is unchanged: mono signed PCM16 at 24 kHz. The server resamples to
  the model's processor rate.
- Keep the client's `create_response=false`. Keep `interrupt_response=false`
  when that is what the client sent. Open WebUI's published tag defaults
  interruption on; the observed operator trace used off. Do not override either
  flag.

After VAD or an explicit commit, the server attaches the original audio, then
schedules recognition. Successful recognition emits
`conversation.item.input_audio_transcription.completed` with `transcript` and
duration `usage`. Failure emits
`conversation.item.input_audio_transcription.failed`. Empty recognition uses
code `transcription_empty`. Existing retirement rules are unchanged; a failed
item does not keep audio by promise. Long committed recordings keep that same
client lifecycle: one original audio item, one completed or failed event, and
the transcript stored on that item. Native audio is not split or replaced by
STT for the client.

`metadata.input_item_id` on the later `response.create` selects that user audio
item and earlier history, keeps injected system context, and excludes later
turns. Assistant `output_audio` and `output_audio_transcript` events are
unchanged.

## HTTP transcription

`POST /v1/audio/transcriptions` uses the same loaded Thinker. Omit `model` to
default to the served alias. Supported fields are plain `json` or `text` plus
`language` and `prompt`. Timestamps, translation, hotwords, beam search, and
`vllm_xargs` are rejected.

```bash
curl http://localhost:8091/v1/audio/transcriptions \
  -F file=@speech.wav \
  -F language=de
```

This endpoint is for direct STT clients. It does not make Open WebUI start a
Realtime answer. Long uploads use the same request and return one aggregate
transcript after every private window finishes, including streamed HTTP.
Short files keep the previous HTTP behavior.

## Optional Chat Completions inline transcription

Ordinary text Chat Completions do not run ASR. Inline recognition is an explicit
opt-in: `stream` must be `true`, and the request must include an
`input_audio_transcription` object (top-level JSON, or the OpenAI SDK
`extra_body` keyword described in [Chat Completions](chat_completions_api.md)).
Optional `model` defaults to `request.model`. The latest user message must
contain exactly one `input_audio` part.

The server first emits a named SSE event, then the normal assistant stream.
These are not historical modality transcript chunks. A stock OpenAI SDK does
not receive those named events; an extension consumer must read the raw SSE.

```bash
curl http://localhost:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "zenith",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "input_audio",
            "input_audio": {"data": "<base64-wav>", "format": "wav"}
          }
        ]
      }
    ],
    "input_audio_transcription": {"language": "de"}
  }'
```

Illustrative first event (`transcript` is an example, not a recorded result):

```
event: omni.input_audio_transcription.completed
data: {"type":"omni.input_audio_transcription.completed","message_index":0,"content_index":0,"transcript":"Guten Tag"}

```

This opt-in does not replace Realtime events for Open WebUI.

## Capability, latency, and limits

Recognition runs after the committed clip is attached. A short clip is one
Thinker pass. A longer clip is split into consecutive private windows using
the loaded model's speech-to-text clip length (about 30 seconds for Qwen3-Omni
by default, not a hardcoded guarantee) and those windows are recognized in
order. Clients still receive one complete transcript or one failure. The
client may then request an answer. Async scheduling does not mean overlapping
GPU work. There is no claim of KV cache reuse, extra concurrency, or free
cost.

Permitted long recordings transcribe completely with the existing Realtime
and HTTP contracts. No extra protocol fields or serving flags are required.
Results are withheld until every private window finishes normally. Any chunk
error, length-limit hit, missing completion, or cancellation fails the whole
job. A digitally silent window may add no words; a window containing any
nonzero samples that returns empty text fails the job. There is no successful
partial prefix, no indefinite streaming transcript, and no frontend
transcript fallback for later response generation.

Encoded HTTP uploads still obey the existing global duration, decoded-byte,
and upload-size limits (defaults 600 s, 256 MiB decoded, 25 MB upload). Those
defaults are configurable; they do not promise every permitted file finishes
inside the parent timeout or deployment resources. A permitted file is
decoded in full, not truncated to the ASR clip length.

| Limit                   | Behavior                                                                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Per Realtime connection | ASR jobs run serially                                                                                                                |
| Pending ASR jobs        | At most 32; further commits fail that item and do not start a response                                                               |
| ASR execution           | 300 s timeout for the whole committed job, including sequential private windows                                                      |
| Session history         | 64 MiB still applies                                                                                                                 |
| Duration                | The loaded model's speech-to-text clip length bounds each private window, not the whole committed file                               |
| Settings                | Captured at commit; later session updates do not rewrite an in-flight job                                                            |
| Cancellation            | Item delete and disconnect cancel that ASR job. Clearing the input buffer or `response.cancel` does not cancel already-committed ASR |
| Identity                | ASR uses distinct `asr-<session>-<item>` ids, not assistant response state                                                           |
| Attachment              | Transcript writes the same user audio part 0; it is never an assistant delta                                                         |
| Response context        | A completed long transcript does not mean the original native audio fits the later answer. `limit_mm_per_prompt.audio` counts native response items, not the private ASR windows. |

## Logging and troubleshooting

ASR uses the existing vLLM module loggers. There is no extra tracing stack,
handler, or serving flag. Optional DEBUG detail uses the same upstream
logging configuration already forwarded by the runtime.

On init, INFO records the served aliases and transcription stage. A
successful Realtime turn then logs:

- INFO `Audio committed` with `session_id`, `item_id`, audio bytes, and
  whether transcription was scheduled
- INFO `[realtime] ASR submit` with `request_id` (`asr-<session>-<item>`)
  and `queue_ms`
- INFO `[realtime] ASR completed` with the same item/request and
  `elapsed_ms` (execution time after the job starts; queue wait is
  separate)

A queued record is DEBUG. Cancellation stays INFO; disconnect cleanup is
DEBUG. Enable DEBUG only when you need audio sample count, duration, source
and target rates, or generate-stage prompt token count and max tokens.

`input_audio_buffer.clear` logs INFO only when the buffer was nonempty:
discarded bytes and whether speech was in progress. That is a client
clear, not mute, and it does not cancel already-committed ASR jobs.

WARNING covers rejected settings, unsupported audio, append and history
limits, a full ASR queue, timeout, and empty recognition. Unexpected
Realtime failures log an exception once with session, item, and request
IDs and a standard traceback. Unexpected inline failures log once with
the request ID and message/content indices. Transcripts, raw payloads,
request objects, and locals are not added. The client still receives the
existing generic transcription failure. A runtime `ValueError` is logged
as an error with traceback so an internal bug is not treated as
validation by type alone.

HTTP settings rejections are WARNING. Ordinary unexpected HTTP ASR
exceptions are logged once with the upstream request-metadata ID when
that ID is available. The existing generic `ValueError` response is
unchanged. `VLLMError` stays with upstream handlers and the existing
engine `log_error_stack` policy; stream exceptions are already logged
there. Do not expect a new trace for every HTTP error.

If you see `ASR completed` and Open WebUI still does not answer, the
provider finished recognition; Open WebUI owns the later `response.create`.
`transcription_scheduled=False` on commit means no ASR job was scheduled;
check session settings, a nearby queue-full warning, and whether the
connection was already closing. It is not proof that transcription is
disabled. If submit never completes, use the request ID plus `queue_ms` /
`elapsed_ms` and any WARNING or exception with that ID.

## Verification still required

The CPU regression modules are marked `core_model` and `cpu`. They use mocks
and do not load weights. With this checkout and the project's test extras
installed, commands that have not been run here:

```bash
pytest tests/entrypoints/openai_api/test_serving_transcription.py -v
pytest tests/entrypoints/openai_api/test_serving_transcription.py \
  -m 'core_model and cpu' --run-level core_model -v
```

Related unrun files:
`tests/entrypoints/openai_api/test_realtime_server_vad.py`,
`tests/entrypoints/openai_api/test_realtime_prompt_preflight.py`, and
`tests/model_executor/models/qwen3_omni/test_qwen3_omni_transcription.py`.

An operator reports working text chat and Realtime voice on this branch.
That is not a substitute for those tests, live syslog of the
request-boundary logs, English and German recognition coverage, or an Open
WebUI cycle in which Open WebUI persists the user and assistant transcripts
while the provider retains the original audio for inference, including
overlap, tools, and cancellation. That cycle does not require Open WebUI to
store the original microphone recording. Long committed recordings that
exceed the loaded model's speech-to-text clip length are implemented in
this tree; they have not been live-validated here.
