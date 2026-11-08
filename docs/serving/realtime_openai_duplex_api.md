# Realtime Audio WebSocket API

This page documents the OpenAI-compatible, turn-based `/v1/realtime` handler.
It is used on deployments without a native duplex runtime and is selected with
`duplex=0` on a duplex server. It is distinct from the vLLM-Omni full-duplex
protocol at `/v1/realtime?duplex=1` and `/v1/duplex`; see
[Full-Duplex WebSocket API](full_duplex_api.md).

The handler supports a subset of the OpenAI Realtime GA API, including streamed
audio input, manual or server VAD turns, text and audio output, and function
calling.

## Quick Start

The endpoint is currently only supported for Qwen3-Omni. To start:

Start vLLM-Omni with port `8000` bound to an address reachable from Docker:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8000 --host 0.0.0.0
```

Then start the docker compose:

```bash
cd examples/online_serving/qwen3_omni/fullduplex-livekit-frontend
VLLM_OMNI_HOST=$VLLM_OMNI_HOST docker compose up --build
```

Where `VLLM_OMNI_HOST` is a network address. If vLLM-Omni is running on
localhost, use `host.docker.internal`.

Open [http://localhost:3000](http://localhost:3000) and begin chatting.

## Protocol

Messages in both directions are JSON text frames. Audio carried inside an event
is base64-encoded raw PCM16.

| Direction | Event | Purpose |
| --------- | ----- | ------- |
| Server to client | `session.created` | Confirms that the WebSocket connection is ready |
| Client to server | `session.update` | Partially updates the session configuration |
| Client to server | `input_audio_buffer.append` | Appends base64 PCM16 audio |
| Client to server | `input_audio_buffer.commit` | Commits the pending audio as a conversation item |
| Client to server | `conversation.item.create` | Adds a text, audio, or image item to the conversation |
| Client to server | `response.create` | Starts a response from the current session and conversation state |
| Server to client | `session.updated` | Returns the effective session after an update |
| Server to client | `response.created` | Starts a response |
| Server to client | `response.output_text.delta` / `.done` | Carries text output |
| Server to client | `response.output_audio_transcript.delta` / `.done` | Carries the transcript of audio output |
| Server to client | `response.output_audio.delta` | Carries incremental PCM16 response audio |
| Server to client | `response.output_audio.done` | Marks the end of response audio |
| Server to client | `response.done` | Completes the response and reports its status and usage |
| Server to client | `error` | Reports an invalid event, model, or audio payload |

A minimal text-input turn is:

```json
{"type":"session.update","session":{"type":"realtime","audio":{"output":{"voice":"ethan","speed":1.0}},"output_modalities":["audio"]}}
{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"What is the capital of France?"}]}}
{"type":"response.create","response":{"audio":{"output":{"voice":"chelsie"}}}}
```

The voice names above are examples from a Qwen3-Omni checkpoint. The
response-level `chelsie` voice overrides the session voice for this response
only. An empty `response.create` uses the session settings without an override.

For an audio-input turn, append one or more chunks, commit the buffer, and then
create the response:

```json
{"type":"input_audio_buffer.append","audio":"<base64-pcm16>"},
{"type":"input_audio_buffer.commit"},
{"type":"response.create"}
```

## Session Configuration

`session.update` is a partial, deep merge: omitted fields keep their current
values, while an explicit `null` clears a value. The server replies with
`session.updated` containing the normalized, effective session. Each
`response.create` resolves its settings from the current session.

### Model selection

Choose the served model on the WebSocket connection, for example
`ws://host:8000/v1/realtime?model=Qwen/Qwen3-Omni-30B-A3B-Instruct`. The query
parameter is optional when the server serves one model. If supplied, it must
match that model; otherwise the server returns an `invalid_request_error` with
`code: "unsupported_model"` and `param: "model"`, then closes the connection.

`session.update` cannot change the connection model. A value in `session.model`
is ignored, and `session.updated` echoes the model chosen at connect time.

### Voice

Set the persistent session voice with `session.audio.output.voice`. Override it
for one response with `response.audio.output.voice`; the response value wins
for that response only and does not change the session voice. Voice names are
matched case-insensitively against the checkpoint's speaker mapping and are
normalized to lower case.

- An unknown name returns `type: "invalid_request_error"`,
  `code: "invalid_value"`, and `param: "session.audio.output.voice"` or
  `param: "response.audio.output.voice"`. The previous session voice remains in
  effect, and a rejected response is not created.
- A custom voice reference such as `{"id": "voice_1234"}` is not supported. It
  returns `param: "voice_id"` with a `null` error code and a `null`
  `error.event_id`.
- If the checkpoint exposes no speaker mapping, names are accepted without
  validation and passed to the model.

### Output speed

`session.audio.output.speed` accepts values from `0.25` through `1.5`,
inclusive. Values outside that range return `code: "decimal_below_min_value"` or
`code: "decimal_above_max_value"`, with `param: "session.audio.output.speed"`;
the previous speed remains in effect. Output speed is session-scoped. A
non-default per-response speed is rejected, and a response-level speed of `1`
does not override the session speed.

Speed is implemented by time-stretching each emitted audio delta independently.
Adjacent deltas are therefore not phase-continuous, and audible clicks or other
artifacts can occur at delta boundaries. A stateful streaming time-stretcher
would be required to remove this limitation; use speed `1.0` when artifact-free
concatenation is required.

### Reasoning

`reasoning.effort` can be set at the session level or overridden for a
response. The effort is honored only for text-only responses. If the effective
`output_modalities` contains `"audio"`, vLLM-Omni ignores the requested effort
and uses the checkpoint template's default because explicitly disabling
thinking corrupts Qwen3-Omni audio generation.

Reasoning support also depends on the model and chat template. To separate
reasoning from client-visible content, start the server with a matching parser,
for example `--reasoning-parser qwen3` for Qwen3-Omni. Without a reasoning
parser, thinking markup may be streamed in response text or an audio transcript.
With a parser, thinking blocks are stripped from both text output and audio
transcripts, but the parser does not make the effort override effective for
audio responses.

### Tools

Function tools require both `--enable-auto-tool-choice` and a matching
`--tool-call-parser`; Qwen3-Omni uses `--tool-call-parser hermes`. Without both,
`response.create` with function tools is rejected. Tool-call markup is removed
from output. Audio output stops once a function call is detected, but audio
deltas already sent to the client cannot be retracted.

Remote MCP tools are not supported. An MCP entry in `tools` or an MCP
`tool_choice` is rejected in both `session.update` and `response.create` with
`code: "unsupported_feature"` and `param: "session.tools"`,
`"session.tool_choice"`, `"response.tools"`, or `"response.tool_choice"`.

### Input processing and turn detection

`audio.input.transcription` and `audio.input.noise_reduction` are accepted on
`session.update` for client compatibility but are not implemented: the server
clears them and does not transcribe input audio or reduce its noise. Non-null
values on `response.create` are rejected. Preprocess audio in the client when
these features are required.

`audio.input.turn_detection` supports `server_vad` and `semantic_vad`;
`semantic_vad` requires `interrupt_response: true`. Set `turn_detection` to
`null` to drive commits manually with `input_audio_buffer.commit` and
`response.create`.

## Audio Handling

- Input and output are mono PCM16 at 24 kHz. Other rates or formats return
  `code: "unsupported_audio_format"`.
- `response.output_audio.delta.delta` contains base64-encoded PCM16 bytes.
  Concatenate the decoded deltas in receive order to construct the output
  waveform; the response audio rate is always 24 kHz.
- `output_modalities: ["audio"]` produces audio and the corresponding
  `response.output_audio_transcript.*` events. The effective modalities come
  from the session and can be overridden by `response.output_modalities`.

## Conversation Items and Image Input

Use `conversation.item.create` to add `input_text` or `input_image` content to
the conversation history. Images do not start a response; send `response.create`
after adding the item. The `input_image.image_url` value is passed through the
model's chat multimodal renderer, which validates and materializes supported
image URLs. Image understanding requires a model with image input support.

## Availability and Limitations

The path is registered on the unified API server, but it is usable only when the
loaded pipeline implements realtime audio input and produces compatible audio
output. Unsupported deployments return an `error` event. The endpoint does not
provide duplex session resume, playback acknowledgement, overlap policy, or
barge-in controls internally. Those must be driven by the client application
via the relevant WS events (for example, `response.cancel` and
`conversation.item.truncate`). Out-of-band responses (`conversation: "none"`)
are not supported.

See the [Qwen3-Omni online serving example](https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/qwen3_omni)
for concurrency options, chunk pacing, and per-delta audio debugging.
