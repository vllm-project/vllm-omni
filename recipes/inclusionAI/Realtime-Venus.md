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
`realtime_venus_omni_duplex.yaml` remains an equivalent explicit alias.

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

The released model can emit `<delegate>...</delegate>` requests. The external
[Realtime-Venus-Harness](https://github.com/inclusionAI/Realtime-Venus/tree/main/harness)
executes these tasks asynchronously and returns `<backend>...</backend>` feedback
to the originating conversation. This adaptation preserves delegate text and
excludes it from speech, but does not integrate the Harness or map delegate spans
to OpenAI `tools` / `tool_calls`. Tool execution and asynchronous result injection
are therefore not available in this serving profile.

## Scope

- Both full-duplex serving (the default) and the explicit turn-based profile
  use the shared Thinker, Talker and Code2Wav stages.
- The external Realtime-Venus delegation harness, asynchronous tool-result
  orchestration and long-video memory adapter are not integrated. Backbone
  compatibility does not establish those external features.
- The Audio-only checkpoint is a separate configuration and is not covered by
  this recipe. Other devices and distributed layouts need separate validation.
- Validation coverage and measured performance are reported in the linked PR
  evidence; basic generation smoke is not a comprehensive quality evaluation.
