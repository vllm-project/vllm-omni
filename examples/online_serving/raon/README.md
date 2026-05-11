# Raon-Speech Online Serving

This directory contains scripts for serving and querying the Raon-Speech model via vLLM Omni's OpenAI-compatible API.

## Installation

Please refer to the [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Launch the Server

```bash
./run_server.sh
```

When launching the server from Docker, allocate enough shared memory for
vLLM-Omni stage communication and PyTorch multiprocessing:

```bash
docker run --gpus all --shm-size=16g ...
```

`--ipc=host` is also acceptable in environments where host IPC sharing is
allowed.

Or manually:

```bash
vllm-omni serve KRAFTON/Raon-Speech-9B \
    --stage-configs-path vllm_omni/model_executor/stage_configs/raon.yaml \
    --host 0.0.0.0 \
    --port 8091 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --omni
```

## Send TTS Requests

### Python Client

```bash
cd examples/online_serving/raon

# Basic TTS
python openai_speech_client.py --text "Hello, world!"

# Voice cloning with a local reference audio
python openai_speech_client.py \
    --text "Hello, this is a cloned voice." \
    --ref-audio /path/to/reference.wav

# Voice cloning with a URL
python openai_speech_client.py \
    --text "Hello, this is a cloned voice." \
    --ref-audio https://example.com/reference.wav

# Save with a specific output path
python openai_speech_client.py \
    --text "Hello, world!" \
    --output my_output.wav
```

### curl

```bash
# Basic TTS
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, how are you?"}' \
    --output output.wav

# Voice cloning (ref_audio as URL)
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is a cloned voice.",
        "ref_audio": "https://example.com/reference.wav"
    }' --output cloned.wav
```

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.audio.speech.create(
    model="KRAFTON/Raon-Speech-9B",
    voice="default",
    input="Hello, how are you?",
)

response.stream_to_file("output.wav")
```

## Streaming Text Input (WebSocket)

The `/v1/audio/speech/stream` WebSocket endpoint accepts text incrementally and generates audio per sentence:

```bash
python streaming_speech_client.py \
    --text "Hello world. How are you? I am fine."

# Simulate STT: send text word-by-word
python streaming_speech_client.py \
    --text "Hello world. How are you? I am fine." \
    --simulate-stt --stt-delay 0.1

# Voice cloning via streaming
python streaming_speech_client.py \
    --text "Hello world. How are you?" \
    --ref-audio /path/to/reference.wav
```

## API Reference

### POST /v1/audio/speech

```json
{
    "input": "Text to synthesize",
    "model": "KRAFTON/Raon-Speech-9B",
    "response_format": "wav",
    "ref_audio": "URL or base64 data URL for voice cloning",
    "ref_text": "Optional transcript of reference audio",
    "max_new_tokens": 2048
}
```

Returns binary audio data.

### Streaming

Set `stream=true` with `response_format="pcm"` to receive raw PCM audio chunks as they are decoded:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "stream": true,
        "response_format": "pcm"
    }' --no-buffer | play -t raw -r 24000 -e signed -b 16 -c 1 -
```

## Notes

- The stage config is at `vllm_omni/model_executor/stage_configs/raon.yaml`.
- Sample rate is 24000 Hz.
- For voice cloning, provide a clear reference audio clip (3-8 seconds recommended).
- Long-form TTS defaults to a final-chunk EOS minimum gate with best-of-k and
  final retry disabled. Override with:
  - `RAON_TTS_LONG_ENABLE_FINAL_BEST_OF_K`
  - `RAON_TTS_LONG_ENABLE_FINAL_EOS_MIN_GATE`
  - `RAON_TTS_LONG_FINAL_EOS_MIN_DURATION_RATIO`
  - `RAON_TTS_LONG_ENABLE_FINAL_EOS_RETRY`
- When running in Docker, use `--shm-size=16g` or `--ipc=host` to avoid shared
  memory exhaustion during concurrent audio generation.
