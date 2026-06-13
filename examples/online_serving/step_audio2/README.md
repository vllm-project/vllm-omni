# Step-Audio2 Online Serving

This directory contains examples for running Step-Audio2 with vLLM-Omni's online serving API.

## Installation

Please refer to [README.md](../../../README.md)

## Launch the Server

```bash
# Async chunk mode (recommended — lower first-packet latency for TTS)
vllm serve stepfun-ai/Step-Audio2-mini --omni --port 8092 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/step_audio_2_async_chunk.yaml \
    --trust-remote-code --enforce-eager
```

Sequential mode:
```bash
vllm serve stepfun-ai/Step-Audio2-mini --omni --port 8092 \
    --stage-configs-path vllm_omni/model_executor/stage_configs/step_audio_2.yaml \
    --trust-remote-code --enforce-eager
```

With local model:
```bash
vllm serve /path/to/Step-Audio-2-mini --omni --port 8092 \
    --trust-remote-code --enforce-eager
```

## Send Requests

### TTS via `/v1/audio/speech` (Recommended)

```bash
cd examples/online_serving/step_audio2

# Python client
python openai_speech_client.py --text "你好世界"

# With custom system prompt
python openai_speech_client.py --text "Hello, how are you?" \
    --instructions "You are a friendly assistant."

# Save to specific file
python openai_speech_client.py --text "你好世界" -o output.wav
```

Or via curl:

```bash
curl -X POST http://localhost:8092/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model":"stepfun-ai/Step-Audio2-mini","input":"你好世界","voice":"default"}' \
    --output output.wav
```

This endpoint bypasses the chat template and directly triggers TTS mode. It supports async chunk streaming for low first-packet latency.

**Note**: Speaker voice is controlled by `STEP_AUDIO2_DEFAULT_PROMPT_WAV` env var on the server side.

### Chat Completions (ASR / S2ST)

```bash
# Audio to Text (ASR)
python openai_chat_completion_client.py --query-type audio_to_text

# Audio to Audio (S2ST)
python openai_chat_completion_client.py --query-type audio_to_audio --audio-path /path/to/input.wav
```

| Argument | Description |
|----------|-------------|
| `--query-type`, `-q` | Query type: `audio_to_text`, `text_to_audio`, `audio_to_audio` |
| `--audio-path`, `-a` | Path to input audio file (local or URL) |
| `--text`, `-t` | Text to synthesize (for TTS mode) |
| `--prompt`, `-p` | Custom prompt/question |
| `--output-dir`, `-o` | Output directory for audio files (default: `output_online`) |
| `--api-base` | API base URL (default: `http://localhost:8092/v1`) |

### Curl (Chat Completions)

```bash
# Audio to Text
bash run_curl.sh audio_to_text

# Text to Audio
bash run_curl.sh text_to_audio

# Audio to Audio
bash run_curl.sh audio_to_audio
```

## Query Types

### 1. Audio to Text (ASR)

Transcribe audio to text.

```bash
python openai_chat_completion_client.py \
    --query-type audio_to_text \
    --audio-path /path/to/speech.wav \
    --prompt "Transcribe this audio."
```

### 2. Text to Audio (TTS)

Convert text to speech:

```bash
# Via speech endpoint (recommended, returns WAV directly)
python openai_speech_client.py --text "Hello, welcome to Step-Audio2."

# Via chat completions
python openai_chat_completion_client.py \
    --query-type text_to_audio \
    --text "Hello, welcome to Step-Audio2."
```

### 3. Audio to Audio (S2ST)

Process input audio and generate text transcription + audio output.

```bash
python openai_chat_completion_client.py \
    --query-type audio_to_audio \
    --audio-path /path/to/source.wav
```

## Output

- **Text output**: Printed to console
- **Audio output**: Saved to `output_online/audio_0.wav` (24kHz WAV)

## API Format

Step-Audio2 uses the OpenAI-compatible chat completions API:

```json
{
  "model": "stepfun-ai/Step-Audio2-mini",
  "messages": [
    {
      "role": "system",
      "content": [{"type": "text", "text": "Transcribe the audio."}]
    },
    {
      "role": "user",
      "content": [
        {"type": "audio_url", "audio_url": {"url": "..."}},
        {"type": "text", "text": "Please transcribe."}
      ]
    }
  ],
  "sampling_params_list": [
    {"temperature": 0.7, "max_tokens": 1024},
    {"temperature": 0.0, "max_tokens": 1}
  ]
}
```

## Performance

### Async Chunk vs Sequential

Benchmark via `/v1/audio/speech` (4x RTX 3090, 10 prompts, concurrency=1):

| Mode | Mean TTFP | Mean E2E | Mean RTF |
|------|-----------|----------|----------|
| Sequential | 4316ms | 4316ms | 0.938 |
| **Async Chunk** | **1437ms** | 4362ms | 0.949 |

Async chunk reduces TTFP by **67%** by streaming audio token chunks from Thinker to Token2Wav as they are generated. RTF < 1 in both modes (real-time capable).

## Troubleshooting

### Server not responding
- Check if the server is running: `curl http://localhost:8092/v1/models`
- Verify the port number matches

### FileNotFoundError: prompt_wav file not found
- Ensure `default_female.wav` exists at `{model_dir}/assets/default_female.wav`
- Or set `STEP_AUDIO2_DEFAULT_PROMPT_WAV` environment variable when launching the server

### Audio not generated
- For TTS, use the `/v1/audio/speech` endpoint (recommended) or `openai_speech_client.py`
- For chat completions TTS, ensure the prompt ends with `<tts_start>`
- Check server logs for errors

### Out of memory
- Reduce `gpu_memory_utilization` in stage configs
- Use a smaller batch size
