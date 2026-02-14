# Qwen3-TTS

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/qwen3_tts>.


## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Supported Models

| Model | Task Type | Description |
|-------|-----------|-------------|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | CustomVoice | Predefined speaker voices with optional style control |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | VoiceDesign | Natural language voice style description |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base | Voice cloning from reference audio |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | CustomVoice | Smaller/faster variant |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | Base | Smaller/faster variant for voice cloning |

## Run examples (Qwen3-TTS)

### Launch the Server

```bash
# CustomVoice model (predefined speakers)
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager

# VoiceDesign model
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager

# Base model (voice cloning)
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

If you have custom stage configs file, launch the server with command below
```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --stage-configs-path /path/to/stage_configs_file \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager
```

Alternatively, use the convenience script:
```bash
./run_server.sh                  # Default: CustomVoice model
./run_server.sh CustomVoice      # CustomVoice model
./run_server.sh VoiceDesign      # VoiceDesign model
./run_server.sh Base             # Base (voice clone) model
```

### Send TTS Request

Get into the example folder
```bash
cd examples/online_serving/qwen3_tts
```

####  Send request via python

```bash
# CustomVoice: Use predefined speaker
python openai_speech_client.py \
    --text "你好，我是通义千问" \
    --voice vivian \
    --language Chinese

# CustomVoice with style instruction
python openai_speech_client.py \
    --text "今天天气真好" \
    --voice ryan \
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

The Python client supports the following command-line arguments:

- `--api-base`: API base URL (default: `http://localhost:8091`)
- `--model` (or `-m`): Model name/path (default: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`)
- `--task-type` (or `-t`): TTS task type. Options: `CustomVoice`, `VoiceDesign`, `Base`
- `--text`: Text to synthesize (required)
- `--voice`: Speaker/voice name (default: `vivian`). Options: `vivian`, `ryan`, `aiden`, etc.
- `--language`: Language. Options: `Auto`, `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, `Italian`
- `--instructions`: Voice style/emotion instructions
- `--ref-audio`: Reference audio file path or URL for voice cloning (Base task)
- `--ref-text`: Reference audio transcript for voice cloning (Base task)
- `--response-format`: Audio output format (default: `wav`). Options: `wav`, `mp3`, `flac`, `pcm`, `aac`, `opus`
- `--output` (or `-o`): Output audio file path (default: `tts_output.wav`)

####  Send request via curl

```bash
# Simple TTS request
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English"
    }' --output output.wav

# With style instruction
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "I am so excited!",
        "voice": "vivian",
        "instructions": "Speak with great enthusiasm"
    }' --output excited.wav

# List available voices in CustomVoice models
curl http://localhost:8091/v1/audio/voices
```

### Using OpenAI SDK

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

### Using Python httpx

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

### FAQ

If you encounter error about backend of librosa, try to install ffmpeg with command below.
```
sudo apt update
sudo apt install ffmpeg
```

## API Reference

### Endpoint

```
POST /v1/audio/speech
Content-Type: application/json
```

This endpoint follows the [OpenAI Audio Speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech) format with additional Qwen3-TTS parameters.

### Voices Endpoint

```
GET /v1/audio/voices
```

Lists available voices for the loaded model:

```json
{
    "voices": ["aiden", "dylan", "eric", "one_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
}
```

### Request Body

```json
{
    "input": "Text to synthesize",
    "voice": "vivian",
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

Returns binary audio data with appropriate `Content-Type` header (e.g., `audio/wav`).

## Parameters

### OpenAI Standard Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | Text to synthesize |
| `model` | string | server's model | Model to use (optional, should match server if specified) |
| `voice` | string | "vivian" | Speaker name (e.g., vivian, ryan, aiden) |
| `response_format` | string | "wav" | Audio format: wav, mp3, flac, pcm, aac, opus |
| `speed` | float | 1.0 | Playback speed (0.25-4.0) |

### vLLM-Omni Extension Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | string | "CustomVoice" | Task: CustomVoice, VoiceDesign, or Base |
| `language` | string | "Auto" | Language (see supported languages below) |
| `instructions` | string | "" | Voice style/emotion instructions |
| `max_new_tokens` | int | 2048 | Maximum tokens to generate |

**Supported languages:** Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

### Voice Clone Parameters (Base task)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ref_audio` | string | **Yes** | Reference audio (URL or base64 data URL) |
| `ref_text` | string | No | Transcript of reference audio (for ICL mode) |
| `x_vector_only_mode` | bool | No | Use speaker embedding only (no ICL) |

## Limitations

- **No streaming**: Audio is generated completely before being returned. Streaming will be supported after the pipeline is disaggregated (see RFC #938).
- **Single request**: Batch processing is not yet optimized for online serving.

## Troubleshooting

1. **TTS model did not produce audio output**: Ensure you're using the correct model variant for your task type (CustomVoice task → CustomVoice model, etc.)
2. **Connection refused**: Make sure the server is running on the correct port
3. **Out of memory**: Use smaller model variant (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`) or reduce `--gpu-memory-utilization`
4. **Unsupported speaker**: Use `/v1/audio/voices` to list available voices for the loaded model
5. **Voice clone fails**: Ensure you're using the Base model variant for voice cloning

## Example materials

??? abstract "openai_speech_client.py"
    ``````py
    --8<-- "examples/online_serving/qwen3_tts/openai_speech_client.py"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/qwen3_tts/run_server.sh"
    ``````
