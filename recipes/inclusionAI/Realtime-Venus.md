# Realtime-Venus Omni

[Realtime-Venus](https://huggingface.co/inclusionAI/Realtime-Venus) adapts
MiniCPM-o 4.5 with a Qwen3-8B language backbone, SigLIP vision encoder,
Whisper audio encoder and S3 speech generation. vLLM-Omni reuses the existing
MiniCPM-o Thinker → Talker → Code2Wav implementation.

## Prepare the checkpoint

The repository contains two checkpoints. Use the **Omni subdirectory**, not
the repository root, as the model path. Its small `config.json` refers to a
separately pinned root YAML; download both together:

```bash
hf download inclusionAI/Realtime-Venus \
  --include 'Realtime-Venus-Omni/*' --include config.yaml \
  --local-dir ./Realtime-Venus
```

For reproducibility, add `--revision` with the model revision you selected.
Keep `config.yaml` beside `Realtime-Venus-Omni/`, including when moving to an
offline host. The native config loader verifies its SHA-256 and model identity
before constructing the nested configs. A missing or mismatched YAML fails
explicitly; the adapter does not download a different configuration at startup.
The checkpoint's tokenizer and processor still require `--trust-remote-code`.

## Serve (full duplex by default)

```bash
vllm serve ./Realtime-Venus/Realtime-Venus-Omni \
  --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/realtime_venus_omni.yaml \
  --host 127.0.0.1 --port 8091
```

The default profile serves continuous audio/video sessions over WebSocket.
Omitting `--deploy-config` selects this same full-duplex profile automatically.

The profile inherits the [MiniCPM-o deployment](../../vllm_omni/deploy/minicpmo_4_5.yaml)
settings, changing only the pipeline identity. This includes five Flow steps
and, on CUDA, FP16 Flow, the default TF32 policy and two Code2Wav replicas on
logical GPU 0. CUDA graphs and Talker multi-frame decode are shared as well.
These are serving defaults, not a claim of speech-quality equivalence to the
upstream 10-step FP32 reference.

Connect to `ws://127.0.0.1:8091/v1/realtime?duplex=1` using the existing
[DuplexClient](../../docs/serving/realtime_duplex_api.md) or
[MiniCPM-o duplex client](../../examples/online_serving/minicpmo/realtime_duplex_demo.py)
with the Venus model path. Use mono PCM16 at 16 kHz for input and 24 kHz for
output, `overlap_policy="listen_only"`, and
`playback_commit_policy="ack_only"`. Report audio actually played with
`playback.ack`; attach camera frames to their corresponding audio units.
Incoming audio continues while the model speaks. The model owns listen/speak
decisions; client-forced destructive cancellation remains subject to the
session's advertised capabilities.

Delegated text remains available in the text stream but is excluded from
speech. A fully delegated unit drains preceding speech once, without closing
the dialogue turn. Further delegated units stay silent; resumed speech starts
with fresh synthesis state, and a terminator-only unit cannot replay the tail.
The turn profile inherits `minicpmo_4_5_turn.yaml`, including its Thinker
kernel and compilation settings; both modes share MiniCPM-o's Flow defaults.

## Turn-based HTTP serving

Select the turn profile explicitly for ordinary `/v1/chat/completions` requests:

```bash
vllm serve ./Realtime-Venus/Realtime-Venus-Omni \
  --omni --trust-remote-code \
  --deploy-config vllm_omni/deploy/realtime_venus_omni_turn.yaml \
  --host 127.0.0.1 --port 8091
```

Text, image, audio and video inputs use the shared multimodal processor.
Request `modalities: ["text", "audio"]` for speech as well as text. Use the
existing [chat-completions clients](../../examples/online_serving/) and
[MiniCPM-o recipe](../OpenBMB/MiniCPM-o-4_5.md) for request formats. Engine mode
is selected at startup; WebSocket query parameters do not switch engines.

## Basic examples

Run these commands from the vLLM-Omni repository root after starting the
corresponding server above. Pass the same model path to the server and client.
The examples reuse the MiniCPM-o clients; no Venus-specific client is needed.
Replace the local image, audio and video paths with your own short inputs.

### Text question with text output

Use the turn server for this and the following four HTTP examples. Disabling
thinking keeps this short request's token budget available for the answer.

```bash
curl http://127.0.0.1:8091/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "./Realtime-Venus/Realtime-Venus-Omni",
    "messages": [{"role": "user", "content": "What is the capital of China?"}],
    "modalities": ["text"],
    "max_tokens": 96,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

### Streaming text and speech

```bash
python examples/online_serving/minicpmo/openai_chat_completion_client_for_multimodal_generation.py \
  --model ./Realtime-Venus/Realtime-Venus-Omni --port 8091 \
  --query-type text --prompt "Say: Hello, welcome." \
  --modalities text,audio --stream
```

The client prints text and saves WAV output. Without `--stream`, it saves
the completed response's audio. Text and audio may occupy separate choices;
clients should inspect every choice for `message.content` and `message.audio`.
Speech output is mono at 24 kHz.

### Image question

```bash
python examples/online_serving/minicpmo/openai_chat_completion_client_for_multimodal_generation.py \
  --model ./Realtime-Venus/Realtime-Venus-Omni --port 8091 \
  --query-type use_image --image-path /path/to/image.jpg \
  --prompt "Describe the main object and its color." --modalities text
```

### Audio question

```bash
python examples/online_serving/minicpmo/openai_chat_completion_client_for_multimodal_generation.py \
  --model ./Realtime-Venus/Realtime-Venus-Omni --port 8091 \
  --query-type use_audio --audio-path /path/to/question.wav \
  --prompt "Answer the question spoken in the audio." --modalities text,audio
```

### Video question

```bash
python examples/online_serving/minicpmo/openai_chat_completion_client_for_multimodal_generation.py \
  --model ./Realtime-Venus/Realtime-Venus-Omni --port 8091 \
  --query-type use_video --video-path /path/to/short-video.mp4 \
  --prompt "Describe what happens in the video." --modalities text
```

This HTTP example sends video frames. Use the duplex example below to send
the video's audio together with its frames.

### Continuous audio/video conversation

Start the default duplex server for this example. It uses the checkpoint's
bundled `speech_in.mp4`, whose audio contains a spoken request. The client
requires FFmpeg for media decoding and streams synchronized audio and frames.

```bash
python examples/online_serving/minicpmo/realtime_duplex_demo.py \
  --url 'ws://127.0.0.1:8091/v1/realtime?duplex=1' \
  --model ./Realtime-Venus/Realtime-Venus-Omni \
  --input-video ./Realtime-Venus/Realtime-Venus-Omni/assets/speech_in.mp4 \
  --tail-silence-s 8 --timeout-s 180 --output-dir ./venus-duplex-output
```

For audio-only input to the **Omni checkpoint**, replace `--input-video` with
`--input-wav /path/to/question-16k-mono.wav` (16 kHz mono PCM16). Inspect the
saved transcript, audio and event results: a native duplex model can choose
to keep listening, so successful transport alone does not prove it answered
the question correctly.

## Tool execution

The default **duplex** server maps a complete `<delegate>...</delegate>` span
to a Realtime `function_call` named `delegate`, with JSON arguments
`{"question": "the model's natural-language task"}`. It accepts one delegation
per model turn, including spans split across audio units. Repeating the same
span while waiting does not execute the tool again. The original span
remains in text output and is excluded from speech.

The application executes the task and returns a `function_call_output` with
the matching `call_id`. The server wraps the result in `<backend>...</backend>`
and waits for the foreground turn to end or the model to return to listening.
A synthetic segment-complete listen does not release the result. The next input
unit then carries it after the media, in the same Thinker KV cache.
If the model repeats the completed delegation while waiting, that control
boundary also admits the result without executing the tool again.
The model can then speak the answer. Input continues while the tool runs;
no `response.create` is needed. The server also schedules a silence continuation
when a result arrives. Applications should keep streaming input or silence so
longer answers have enough model time.

`capabilities.tool_mode` is `delegate_only` and
`function_call_output_continuation` is `automatic`. Venus does not select among
arbitrary named functions: dispatch the natural-language question in your
`delegate` handler, or forward it to your own service. The server never executes
application code. The external
[Realtime-Venus-Harness](https://github.com/inclusionAI/Realtime-Venus/tree/main/harness)
is optional application infrastructure, not a serving dependency.

The shared [tool lifecycle](../../docs/serving/realtime_duplex_api.md#tool-calls-results-deadlines-and-cancellation)
handles unknown/duplicate results, deadlines, cancellation and late replies.
Results are limited to 4000 characters, with at most eight pending results and
4096 pending tokens per session. Reserved token spellings quoted in a result
are encoded as ordinary text; only the server's outer `<backend>` delimiters
act as control tokens. Rejected results leave the call pending for a
corrected reply. Cancellation retires results not yet submitted to the model;
submitted work follows the existing epoch fence. The CLI below keeps pending
calls across interruptions; custom clients choose their interruption policy.

### Prepare two local tools

Start the duplex server above. In a separate working directory, create a
small inventory fixture and an async handler. These examples use local data;
they do not contact an external service.

```bash
printf '{"widget": 73}\n' > inventory.json
cat > tool_handlers.py <<'PY'
import asyncio
import json
import re
from pathlib import Path

async def delegate(arguments):
    question = arguments["question"]
    if "inventory" in question.lower() or "库存" in question:
        inventory = json.loads(await asyncio.to_thread(Path("inventory.json").read_text))
        return json.dumps({"widget_inventory": inventory["widget"]})
    numbers = re.findall(r"-?\d+", question)
    if len(numbers) == 2 and any(word in question.lower() for word in ("add", "sum", "+", "加")):
        return json.dumps({"sum": int(numbers[0]) + int(numbers[1])})
    raise ValueError("This demo handles inventory lookup and adding two integers only")

async def delayed_delegate(arguments):
    await asyncio.sleep(0.5)
    return await delegate(arguments)
PY
cat > tools.json <<'JSON'
[{"type":"function","name":"delegate","description":"Delegate a task to the application",
  "parameters":{"type":"object","properties":{"question":{"type":"string"}},
  "required":["question"]}}]
JSON
```

### Add two numbers

This seeds the native session with text and streams silence. It is not a
text-only `conversation.item.create` turn, which native MiniCPM/Venus rejects.
Use the same model path as the running server:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python -m vllm_omni.clients.tool_calls \
  --url 'ws://127.0.0.1:8091/v1/realtime?duplex=1' \
  --model ./Realtime-Venus/Realtime-Venus-Omni \
  --tools-json tools.json --handler delegate=tool_handlers:delegate \
  --initial-user-text 'Ask the backend to add 12 and 30. Delegate the calculation instead of doing it yourself.' \
  --timeout 60 --output-wav answer.wav --ack-collected-audio
```

### Look up local inventory

Run the same command with this text seed and greedy sampling:

```bash
--initial-user-text 'Please delegate a lookup of the current widget inventory to the backend. Do not guess the count.' \
--temperature 0
```

The handler reads `inventory.json`; the returned count is 73. Changing that
file changes the tool's result without changing the prompt. Inspect the emitted
`response.function_call_arguments.done`, the accepted `function_call_output`
item, and the subsequent answer. A successful connection alone is not proof
that the model delegated or used the result.

### Check a delayed tool result

Use the delayed handler to keep input streaming while the lookup is pending:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" \
python -m vllm_omni.clients.tool_calls \
  --url 'ws://127.0.0.1:8091/v1/realtime?duplex=1' \
  --model ./Realtime-Venus/Realtime-Venus-Omni \
  --tools-json tools.json --handler delegate=tool_handlers:delayed_delegate \
  --initial-user-text 'Please delegate a lookup of the current widget inventory to the backend. Do not guess the count.' \
  --temperature 0 --timeout 60 --output-wav delayed-answer.wav --ack-collected-audio
```

Check that one call receives one accepted result, the answer after acceptance
contains `73`, and that response emits nonempty audio and completes in the same
session without protocol errors. Preserve the event log, checkpoint revision,
server commit and deployment settings when reporting the result.

The public events do not expose the internal pending-result queue. The CPU
regression below separately checks that an accepted result remains queued
through an ordinary segment completion, reaches exactly one append at a model
listen, turn-end or repeated-delegation boundary, and cannot be replayed by a
duplicate result. Run it alongside the real-weight smoke when changing these
boundaries:

```bash
python -m pytest -q \
  tests/engine/duplex/test_session_runner_venus_tools.py::test_result_reaches_one_append_and_duplicate_cannot_replay_it
```

### Ask by voice

Record “Please delegate a lookup of the widget inventory to the backend” as
mono PCM16 at 16 kHz. In the addition command, replace
`--initial-user-text ...` with:

```bash
--input-wav /path/to/inventory-question.wav
```

The generic CLI prints JSON events, requests text and audio for native input,
and saves speech with `--output-wav`. Realtime sessions require playback ACKs.
The explicit `--ack-collected-audio` option simulates immediate playback for
this offline probe by acknowledging received audio; it does not play sound.
This also lets the model continue after a spoken prelude to a tool call.
In an interactive application, run
`DuplexToolRunner(client, {"delegate": delegate}).run()` alongside your
audio playback loop, use `playback_commit_policy="ack_only"`, and acknowledge
audio actually played. Handler failures send
`function_call.cancel` with reason `failed`; they are not returned as invented
business results. Stop the runner when closing the session so in-flight handlers
are cancelled.

This tool protocol is for the duplex Realtime endpoint. The explicit turn
profile still serves ordinary multimodal HTTP requests; it does not implement
OpenAI Chat Completions `tool_calls` for Venus.

## Scope

- Both full-duplex serving (the default) and the explicit turn-based profile
  use the shared Thinker, Talker and Code2Wav stages.
- Application tools use the shared Realtime lifecycle. The external
  Realtime-Venus Harness and long-video memory adapter are not bundled.
- The Audio-only checkpoint is a separate configuration and is not covered by
  this recipe. Other devices and distributed layouts need separate validation.
- Validation coverage and measured performance are reported in the linked PR
  evidence; basic generation smoke is not a comprehensive quality evaluation.
