# Speech API

vLLM-Omni provides an OpenAI-compatible API for text-to-speech (TTS) generation. Supported TTS models include:

- **Qwen3-TTS** (`Qwen/Qwen3-TTS-12Hz-*`) -- Qwen3-based TTS with CustomVoice, VoiceDesign, and Base (voice cloning) task types. Output: 24 kHz.
- **Fish Speech S2 Pro** (`fishaudio/s2-pro`) -- Dual-AR TTS with DAC codec. Supports text-to-speech and voice cloning via reference audio. Output: 44.1 kHz.

Each server instance runs a single model (specified at startup via `vllm serve <model> --omni`).

## Quick Start

### Start the Server

```bash
# Qwen3-TTS: CustomVoice model (predefined speakers)
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager

# Fish Speech S2 Pro
vllm-omni serve fishaudio/s2-pro \
    --stage-configs-path vllm_omni/model_executor/stage_configs/fish_speech_s2_pro.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager \
    --gpu-memory-utilization 0.9
```

### Generate Speech

**Using curl:**

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English"
    }' --output output.wav
```

**Using Python:**

```python
import httpx

response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English",
    },
    timeout=300.0,
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

**Using OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")

response = client.audio.speech.create(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    voice="vivian",
    input="Hello, how are you?",
)

response.stream_to_file("output.wav")
```

## API Reference

### Endpoint

```
POST /v1/audio/speech
Content-Type: application/json
```

### Request Parameters

#### OpenAI Standard Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | The text to synthesize into speech |
| `model` | string | server's model | Model to use (optional, should match server if specified) |
| `voice` | string | "vivian" | Speaker name (e.g., vivian, ryan, aiden) |
| `response_format` | string | "wav" | Audio format: wav, mp3, flac, pcm, aac, opus |
| `speed` | float | 1.0 | Playback speed (0.25-4.0) |

#### vLLM-Omni Extension Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | string | "CustomVoice" | TTS task type: CustomVoice, VoiceDesign, or Base |
| `language` | string | "Auto" | Language (see supported languages below) |
| `instructions` | string | "" | Voice style/emotion instructions |
| `max_new_tokens` | integer | 2048 | Maximum tokens to generate |
| `initial_codec_chunk_frames` | integer | null | Per-request initial chunk size override for TTFA tuning. When null, IC is computed dynamically based on server load. |

**Supported languages:** Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

#### Voice Clone Parameters (Base task)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ref_audio` | string | null | Reference audio (HTTP URL, base64 data URL, or `file://` URI with `--allowed-local-media-path`) |
| `ref_text` | string | null | Transcript of reference audio |
| `x_vector_only_mode` | bool | null | Use speaker embedding only (no ICL) |

### Response Format

Returns binary audio data with appropriate `Content-Type` header (e.g., `audio/wav`).

### Voices Endpoint

```
GET /v1/audio/voices
```

Lists available voices for the loaded model.

```json
{
    "voices": ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
}
```
```
POST /v1/audio/voices
Content-Type: multipart/form-data
```

Upload a new voice sample for voice cloning in Base task TTS requests.

**Form Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `audio_sample` | file | Yes | Audio file (max 10MB, supported formats: wav, mp3, flac, ogg, aac, webm, mp4) |
| `consent` | string | Yes | Consent recording ID |
| `name` | string | Yes | Name for the new voice |

**Response Example:**

```json
{
  "success": true,
  "voice": {
    "name": "custom_voice_1",
    "consent": "user_consent_id",
    "created_at": 1738660000,
    "mime_type": "audio/wav",
    "file_size": 1024000
  }
}
```

**Usage Example:**

```bash
curl -X POST http://localhost:8091/v1/audio/voices \
  -F "audio_sample=@/path/to/voice_sample.wav" \
  -F "consent=user_consent_id" \
  -F "name=custom_voice_1"
```

## Streaming Text Input (WebSocket)

The `/v1/audio/speech/stream` WebSocket endpoint accepts text incrementally and generates audio per sentence as boundaries are detected.

> Note: text input is always streamed incrementally. Audio output remains sentence-scoped:
> use `stream_audio=false` for one binary frame per sentence, or `stream_audio=true` for one or more PCM chunks per sentence.

### WebSocket Protocol

Client -> Server:

| Message | Description |
|---------|-------------|
| `{"type": "session.config", ...}` | Session configuration (sent once, first message) |
| `{"type": "input.text", "text": "..."}` | Text chunk |
| `{"type": "input.done"}` | End of input, flushes remaining buffer |

Server -> Client:

| Message | Description |
|---------|-------------|
| `{"type": "audio.start", "sentence_index": 0, "sentence_text": "...", "format": "pcm", "sample_rate": 24000}` | Audio generation starting for a sentence |
| Binary frame | Raw audio bytes (one or more PCM chunks when `stream_audio=true`) |
| `{"type": "audio.done", "sentence_index": 0, "total_bytes": 96000, "error": false}` | Audio complete for a sentence |
| `{"type": "session.done", "total_sentences": N}` | Session complete |
| `{"type": "error", "message": "..."}` | Non-fatal error |

### Session Config Parameters

All REST API parameters are supported, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `stream_audio` | bool | false | Stream one or more PCM chunks per sentence over WebSocket |
| `split_granularity` | string | "sentence" | Text splitting granularity |


```bash
DELETE /v1/audio/voices/{name}
```

Delete an uploaded voice sample.

**Path Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `name` | string | Yes | Name of the voice to delete |

**Response Example:**

```json
{
  "success": true,
  "message": "Voice 'custom_voice_1' deleted successfully"
}
```

**Error Response (404 Not Found):**

```json
{
  "success": false,
  "error": "Voice 'unknown_voice' not found"
}
```

**Usage Example:**

```bash
curl -X DELETE http://localhost:8091/v1/audio/voices/custom_voice_1
```

## Examples

### CustomVoice with Style Instruction

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "I am so excited!",
        "voice": "vivian",
        "instructions": "Speak with great enthusiasm"
    }' --output excited.wav
```

### VoiceDesign (Natural Language Voice Description)

```bash
# Start server with VoiceDesign model first
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello world",
        "task_type": "VoiceDesign",
        "instructions": "A warm, friendly female voice with a gentle tone"
    }' --output designed.wav
```

### Base (Voice Cloning)

```bash
# Start server with Base model first
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is a cloned voice",
        "task_type": "Base",
        "ref_audio": "https://example.com/reference.wav",
        "ref_text": "Original transcript of the reference audio"
    }' --output cloned.wav
```

upload voice
```bash
curl -X POST http://localhost:8091/v1/audio/voices \
  -F "audio_sample=@/path/to/voice_sample.wav" \
  -F "consent=user_consent_id" \
  -F "name=custom_voice_1"
```

use upload voice
```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is a cloned voice",
        "task_type": "Base",
        "voice": "custom_voice_1"
    }' --output cloned.wav
```

## Supported Models

### Qwen3-TTS

| Model | Task Type | Description |
|-------|-----------|-------------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | CustomVoice | Predefined speaker voices with optional style control |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | VoiceDesign | Natural language voice style description |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base | Voice cloning from reference audio |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | CustomVoice | Smaller/faster variant |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Base | Smaller/faster variant for voice cloning |

### Fish Speech S2 Pro

| Model | Description |
|-------|-------------|
| `fishaudio/s2-pro` | 4B dual-AR TTS with DAC codec (44.1 kHz). Supports text-to-speech and voice cloning. |

Fish Speech uses `ref_audio` and `ref_text` for voice cloning (no `task_type` needed). The `voice` field should be set to `"default"`. See the [Fish Speech online serving example](../user_guide/examples/online_serving/fish_speech.md) for details.

## Error Responses

### 400 Bad Request

Invalid parameters:

```json
{
    "error": {
        "message": "Input text cannot be empty",
        "type": "BadRequestError",
        "param": null,
        "code": 400
    }
}
```

### 404 Not Found

Model not found:

```json
{
    "error": {
        "message": "The model `xxx` does not exist.",
        "type": "NotFoundError",
        "param": "model",
        "code": 404
    }
}
```

## Troubleshooting

### "TTS model did not produce audio output"

Ensure you're using the correct model variant for your task type:
- CustomVoice task → CustomVoice model
- VoiceDesign task → VoiceDesign model
- Base task → Base model

### Server Not Running

```bash
# Check if server is responding
curl http://localhost:8091/v1/audio/voices
```

### Out of Memory

If you encounter OOM errors:
1. Use smaller model variant: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
2. Reduce `--gpu-memory-utilization`

### Unsupported Speaker

Use `/v1/audio/voices` to list available voices for the loaded model.

## Development

Enable debug logging:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager \
    --uvicorn-log-level debug
```
