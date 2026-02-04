# Qwen3-TTS

This directory contains examples for running Qwen3-TTS models with vLLM-Omni's online serving API.

## Supported Models

| Model | Task Type | Description |
|-------|-----------|-------------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | CustomVoice | Predefined speaker voices with optional style control |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | VoiceDesign | Natural language voice style description |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base | Voice cloning from reference audio |

## Quick Start

### 1. Start the Server

```bash
# CustomVoice model (default)
./run_server.sh

# Or specify task type
./run_server.sh CustomVoice
./run_server.sh VoiceDesign
./run_server.sh Base
```

### 2. Run the Client

```bash
# CustomVoice: Use predefined speaker
python openai_speech_client.py \
    --text "你好，我是通义千问" \
    --voice Vivian \
    --language Chinese

# CustomVoice with style instruction
python openai_speech_client.py \
    --text "今天天气真好" \
    --voice Ryan \
    --instructions "用开心的语气说"

# VoiceDesign: Describe the voice style
python openai_speech_client.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --task-type VoiceDesign \
    --text "哥哥，你回来啦" \
    --instructions "体现撒娇稚嫩的萝莉女声，音调偏高"

# Base: Voice cloning
python openai_speech_client.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --task-type Base \
    --text "Hello, this is a cloned voice" \
    --ref-audio /path/to/reference.wav \
    --ref-text "Original transcript of the reference audio"
```

### 3. Using curl

```bash
# Simple TTS request
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "Vivian",
        "language": "English"
    }' --output output.wav

# With style instruction
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "I am so excited!",
        "voice": "Vivian",
        "instructions": "Speak with great enthusiasm"
    }' --output excited.wav

# List available voices in CustomVoice models
curl http://localhost:8000/v1/audio/voices
```

## API Reference

### Endpoint

```
POST /v1/audio/speech
```

This endpoint follows the [OpenAI Audio Speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech) format with additional Qwen3-TTS parameters.

### Request Body

```json
{
    "input": "Text to synthesize",
    "voice": "Vivian",
    "response_format": "wav",
    "task_type": "CustomVoice",
    "language": "Auto",
    "instructions": "Optional style instructions",
    "ref_audio": "URL or base64 for voice cloning",
    "ref_text": "Reference audio transcript",
    "x_vector_only_mode": false,
    "max_new_tokens": 2048
}
```

> **Note:** The `model` field is optional when serving a single model, as the server already knows which model is loaded.

### Response

Returns audio data in the requested format (default: WAV).

## Parameters

### Standard OpenAI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | required | Text to synthesize |
| `voice` | string | "Vivian" | Speaker/voice name |
| `response_format` | string | "wav" | Audio format: wav, mp3, flac, pcm, aac, opus |
| `speed` | float | 1.0 | Playback speed (0.25-4.0) |
| `model` | string | optional | Model name (optional when serving single model) |

### Qwen3-TTS Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | string | "CustomVoice" | Task: CustomVoice, VoiceDesign, or Base |
| `language` | string | "Auto" | Language: Auto, Chinese, English, Japanese, Korean |
| `instructions` | string | "" | Voice style/emotion instructions |
| `max_new_tokens` | int | 2048 | Maximum tokens to generate |

### Voice Clone Parameters (Base task)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ref_audio` | string | Yes* | Reference audio (file path, URL, or base64) |
| `ref_text` | string | No | Transcript of reference audio (for ICL mode) |
| `x_vector_only_mode` | bool | false | Use speaker embedding only (no ICL) |

## Python Usage

```python
import httpx

# Simple request
response = httpx.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "input": "Hello world",
        "voice": "Vivian",
    },
    timeout=300.0,
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Limitations

- **No streaming**: Audio is generated completely before being returned. Streaming will be supported after the pipeline is disaggregated (see RFC #938).
- **Single request**: Batch processing is not yet optimized for online serving.

## Troubleshooting

1. **Connection refused**: Make sure the server is running on the correct port
2. **Out of memory**: Reduce `--gpu-memory-utilization` in run_server.sh
3. **Unsupported speaker**: Check supported speakers via model documentation
4. **Voice clone fails**: Ensure you're using the Base model variant for voice cloning
