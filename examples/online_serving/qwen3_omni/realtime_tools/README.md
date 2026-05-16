# Qwen3-Omni Realtime API with Tool Calling

This example demonstrates the OpenAI-compatible `/v1/realtime` WebSocket API
with tool calling support for Qwen3-Omni.

The server accepts streamed PCM audio, detects tool calls in the model's
response, emits OpenAI-style function-call events, accepts tool results from
the client, and produces speech audio incorporating the result.

## Supported events

**Client → Server:**

| Type | Purpose |
| --- | --- |
| `session.update` | Configure model, instructions, and tool definitions |
| `input_audio_buffer.append` | Send a chunk of base64-encoded 16 kHz mono PCM16 audio |
| `input_audio_buffer.commit` | Trigger generation on the buffered audio |
| `conversation.item.create` | Send a `function_call_output` tool result back to the model |

**Server → Client:**

| Type | Purpose |
| --- | --- |
| `session.created` | Connection established |
| `conversation.item.input_audio_transcription.delta` / `.done` | Streaming transcript of the model's spoken response (plain / instructions modes) |
| `response.function_call_arguments.delta` / `.done` | Tool call name + JSON arguments |
| `response.audio.delta` | Base64-encoded 24 kHz PCM16 chunk |
| `response.audio.done` | Audio response complete |
| `error` | Error event (for unsupported configurations or runtime errors) |

## `session.update` payload

`session.update` carries the served model name and the session config
(instructions, tool definitions, and optional talker voice):

```json
{
  "type": "session.update",
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "session": {
    "instructions": "You are a helpful voice assistant. Use tools when appropriate.",
    "tools": [ /* see Tool definition format below */ ],
    "voice": "ethan"
  }
}
```

`voice` is optional. Qwen3-Omni supports `"ethan"` (default), `"chelsie"`,
and `"aiden"`. Omitting `voice` falls back to the model's default voice.

## Tool definition format

Tool definitions follow OpenAI's JSON schema format and are passed in
`session.update` under `session.tools`:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name, e.g. Paris"},
                },
                "required": ["location"],
            },
        },
    },
]
```

## Tool call wire format

Qwen3-Omni emits tool calls in its native XML-wrapped JSON format. The server
parses this format and translates it to OpenAI-style streaming events for the
client. From the client's perspective, you only see standard OpenAI events;
the underlying model output looks like:

```
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "Paris"}}
</tool_call>
```

## Tool result format

After receiving a `response.function_call_arguments.done` event, the client
executes the tool and sends back the result with the matching `call_id`:

```json
{
  "type": "conversation.item.create",
  "item": {
    "type": "function_call_output",
    "call_id": "call_abc123",
    "output": "{\"temperature\": 18, \"condition\": \"Sunny\"}"
  }
}
```

The tool result is appended to the session's conversation history. The
server then runs a follow-up audio pass with the full context
(instructions + history including the new tool result) and the model
decides how to respond — paraphrase the result, ask a clarifying
question, chain another tool call, etc. Tool-call detection stays active
on the follow-up pass, so chained tool calls within a single user turn
work the same way.

## Generation modes

The server picks one of three modes per session based on what's set in
`session.update`:

| Mode | Configured | Generation path |
| --- | --- | --- |
| Plain | no tools, no instructions | streaming text pass with multi-turn history |
| Instructions | `instructions` only | streaming text pass with system prompt + history |
| Tools | `tools` set | single audio pass with tool-call detection / abort |

All three modes share the same multi-turn machinery: a side-channel ASR
pass transcribes each user turn in parallel with the response generation
(stage-0 thinker continuous-batches the two), and the resulting transcript
is appended to conversation history so the next turn's prompt has a real
user line instead of a placeholder.

## Multi-turn flow (tools mode)

1. Client sends `session.update` with tools and instructions.
2. Client streams user audio via `input_audio_buffer.append` and commits.
3. Server runs the audio pass. While the thinker is generating, it
   monitors the output for `<tool_call>` markers:
   - If a tool call is detected, the server aborts the engine before the
     talker spins on the XML and emits
     `response.function_call_arguments.delta`/`.done`.
   - Otherwise the talker / code2wav stages stream
     `response.audio.delta` chunks.
4. On the tool path, the client executes the tool and replies with
   `conversation.item.create` (`function_call_output`).
5. Server runs a follow-up audio pass with the tool result appended to
   history. The model decides how to respond — paraphrase the result,
   ask a clarifying question, chain another tool call, etc. Chained tool
   calls within a single turn re-enter step 3.
6. Conversation history (user transcript, tool calls, tool results,
   assistant responses) is retained for follow-up turns within the same
   WebSocket session.

## Running the example

Start the server with the bundled Qwen3-Omni deploy config:

```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091 \
    --deploy-config vllm_omni/deploy/qwen3_omni_moe.yaml
```

The deploy config ships with a conservative stage-0 `gpu_memory_utilization`
that fits on 80 GB GPUs (H100, A100). If you have a larger card or want a
bigger KV-cache budget, copy the yaml and bump that field.

Run the example client (registers a real ``get_current_weather`` tool plus a
similarly-named ``get_city_timezone`` *trap* tool — useful for verifying the
model picks the right tool for a weather question rather than the
plausible-but-wrong one):

```bash
python examples/online_serving/qwen3_omni/realtime_tools/realtime_tools_client.py \
    --url ws://localhost:8091/v1/realtime \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --input-wav ask_weather.wav \
    --output-wav response.wav
```

Pass `--voice` to select a talker voice (`ethan`, `chelsie`, or `aiden`):

```bash
python examples/online_serving/qwen3_omni/realtime_tools/realtime_tools_client.py \
    --voice chelsie \
    --input-wav ask_weather.wav --output-wav response.wav
```

`--input-wav` accepts multiple files to run sequential turns over a single
WebSocket session, demonstrating that conversation context is retained
across turns:

```bash
python examples/online_serving/qwen3_omni/realtime_tools/realtime_tools_client.py \
    --input-wav greeting.wav weather_paris.wav weather_london.wav \
    --output-wav response.wav   # writes response_turn1.wav, _turn2.wav, _turn3.wav
```

Input WAVs must be mono 16-bit PCM at 16 kHz.

## Limitations

- Conversation history grows with each turn within a session; very long
  sessions or large tool responses may approach the smallest stage's
  `max_model_len`. The server logs a warning when the estimated prompt
  size exceeds 50% of that limit (codec generation needs the remaining
  budget — past it, the talker / code2wav stage may stall or truncate).
- The side-channel ASR pass for user-turn transcription runs in parallel
  with the response generation but is best-effort: it has a 2 s wait
  budget at the start of the audio pass and a 3 s budget at the start of
  the next turn. If it hasn't completed by then (e.g. unusually long user
  audio under heavy load), that turn's user item is rendered with a
  placeholder, which can momentarily break Qwen3's user/assistant
  alternation in the next turn's prompt.
