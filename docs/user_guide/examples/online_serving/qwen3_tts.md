# Qwen3-TTS

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/qwen3_tts>.


## 🛠️ Installation

Please refer to [README.md](https://github.com/vllm-project/vllm-omni/tree/main/README.md)

## Supported Models

| Model                                  | Task Type   | Description                                           |
| -------------------------------------- | ----------- | ----------------------------------------------------- |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | CustomVoice | Predefined speaker voices with optional style control |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | VoiceDesign | Natural language voice style description              |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base`        | Base        | Voice cloning from reference audio                    |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | CustomVoice | Smaller/faster variant                                |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base`        | Base        | Smaller/faster variant for voice cloning              |

## Gradio Demo

!!! note "Gradio is an optional dependency"
    The Gradio demo requires the `[demo]` extras. Install them first:

    ```bash
    pip install 'vllm-omni[demo]'
    ```

    Or, if installing from source: `pip install -e '.[demo]'`

Two interactive Gradio demos are available, both supporting all 3 task types:

| Demo     | File                     | Transport    | Streaming Quality                                  |
| -------- | ------------------------ | ------------ | -------------------------------------------------- |
| Standard | `gradio_demo.py`         | HTTP chunked | May have small gaps between chunks                 |
| FastRTC  | `gradio_fastrtc_demo.py` | WebRTC       | Gapless streaming (requires `pip install fastrtc`) |

```bash
# Option 1: Launch server + Standard Gradio together
./run_gradio_demo.sh                                # CustomVoice (default)
./run_gradio_demo.sh --task-type VoiceDesign        # VoiceDesign
./run_gradio_demo.sh --task-type Base               # Voice cloning

# Option 2: If server is already running
python gradio_demo.py --api-base http://localhost:8000

# Option 3: FastRTC demo (gapless streaming)
pip install fastrtc
python gradio_fastrtc_demo.py --api-base http://localhost:8000
```

Then open http://localhost:7860 in your browser.

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
    --speaker vivian \
    --language Chinese

# CustomVoice with style instruction
python openai_speech_client.py \
    --text "今天天气真好" \
    --speaker ryan \
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
- `--speaker`: Speaker name (default: `vivian`). Options: `vivian`, `ryan`, `aiden`, etc.
- `--language`: Language. Options: `Auto`, `Chinese`, `English`, `Japanese`, `Korean`, `German`, `French`, `Russian`, `Portuguese`, `Spanish`, `Italian`
- `--instructions`: Voice style/emotion instructions
- `--ref-audio`: Reference audio file path or URL for voice cloning (Base task). Local paths are automatically base64-encoded by the client before sending to the server.
- `--ref-text`: Reference audio transcript for voice cloning (Base task).
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

### Voices Endpoint

#### GET /v1/audio/voices

List all available voices/speakers from the loaded model, including both built-in model voices and uploaded custom voices.

**Response Example:**
```json
{
  "voices": ["vivian", "ryan", "custom_voice_1"],
  "uploaded_voices": [
    {
      "name": "custom_voice_1",
      "consent": "user_consent_id",
      "created_at": 1738660000,
      "file_size": 1024000,
      "mime_type": "audio/wav",
      "ref_text": "The exact transcript of the audio sample.",
      "speaker_description": "warm narrator"
    }
  ]
}
```

Fields `ref_text` and `speaker_description` are omitted per-entry when not provided at upload time.

#### POST /v1/audio/voices

Upload a new voice sample for voice cloning in Base task TTS requests.

**Form Parameters:**
- `audio_sample` (required): Audio file (max 10MB, supported formats: wav, mp3, flac, ogg, aac, webm, mp4)
- `consent` (required): Consent recording ID
- `name` (required): Name for the new voice
- `ref_text` (optional): Transcript of the audio. Enables in-context voice cloning (higher quality).
- `speaker_description` (optional): Free-form description of the voice (e.g. "warm narrator", "energetic presenter"). Stored as metadata.

**Response Example:**
```json
{
  "success": true,
  "voice": {
    "name": "custom_voice_1",
    "consent": "user_consent_id",
    "created_at": 1738660000,
    "mime_type": "audio/wav",
    "file_size": 1024000,
    "ref_text": "The exact transcript of the audio sample.",
    "speaker_description": "warm narrator"
  }
}
```

Fields `ref_text` and `speaker_description` are omitted when not provided at upload time.

**Usage Example:**
```bash
curl -X POST http://localhost:8091/v1/audio/voices \
  -F "audio_sample=@/path/to/voice_sample.wav" \
  -F "consent=user_consent_id" \
  -F "name=custom_voice_1" \
  -F "ref_text=The exact transcript of the audio sample." \
  -F "speaker_description=warm narrator"
```

### Endpoint

```
POST /v1/audio/speech
Content-Type: application/json
```

This endpoint follows the [OpenAI Audio Speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech) format with additional Qwen3-TTS parameters.

### Request Body

```json
{
    "input": "Text to synthesize",
    "voice": "vivian",
    "response_format": "wav",
    "task_type": "CustomVoice",
    "language": "Auto",
    "instructions": "Optional style instructions",
    "ref_audio":  "HTTP URL, base64 data URL, or file:// URI for voice cloning",
    "ref_text": "Reference audio transcript",
    "x_vector_only_mode": false,
    "max_new_tokens": 2048
}
```

> **Note:** The `model` field is optional when serving a single model, as the server already knows which model is loaded.

### Response

Returns binary audio data with appropriate `Content-Type` header (e.g., `audio/wav`).

### Voice and language (summary)

- **Speaker**: Use the `voice` request field to select the speaker (e.g., `vivian`, `ryan`, `aiden`). List available speakers with `GET /v1/audio/voices`.
- **Language**: Use the `language` field for the codec language tag (`Auto`, `Chinese`, `English`, etc.). Default is `Auto` for automatic detection.
- **CustomVoice**: Requires a valid `voice` from the model’s speaker set. **VoiceDesign**: Use `instructions` to describe the voice. **Base**: Use `ref_audio` and `ref_text` for voice cloning.

## Parameters

### OpenAI Standard Parameters

| Parameter         | Type   | Default        | Description                                                 |
| ----------------- | ------ | -------------- | ----------------------------------------------------------- |
| `input`           | string | **required**   | Text to synthesize                                          |
| `model`           | string | server's model | Model to use (optional, should match server if specified)   |
| `voice`           | string | "vivian"       | Speaker name (e.g., vivian, ryan, aiden)                    |
| `response_format` | string | "wav"          | Audio format: wav, mp3, flac, pcm, aac, opus                |
| `speed`           | float  | 1.0            | Playback speed (0.25-4.0, not supported with `stream=true`) |

### vLLM-Omni Extension Parameters

| Parameter                    | Type   | Default       | Description                                                                                                          |
| ---------------------------- | ------ | ------------- | -------------------------------------------------------------------------------------------------------------------- |
| `task_type`                  | string | "CustomVoice" | Task: CustomVoice, VoiceDesign, or Base                                                                              |
| `language`                   | string | "Auto"        | Language (see supported languages below)                                                                             |
| `instructions`               | string | ""            | Voice style/emotion instructions                                                                                     |
| `max_new_tokens`             | int    | 2048          | Maximum tokens to generate                                                                                           |
| `initial_codec_chunk_frames` | int    | null          | Per-request initial chunk size override for TTFA tuning. When null, IC is computed dynamically based on server load. |
| `stream`                     | bool   | false         | Stream raw PCM chunks as they are decoded (requires `response_format="pcm"`)                                         |

**Supported languages:** Auto, Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

### Voice Clone Parameters (Base task)

| Parameter            | Type   | Required | Description                                                                                     |
| -------------------- | ------ | -------- | ----------------------------------------------------------------------------------------------- |
| `ref_audio`          | string | **Yes**  | Reference audio (HTTP URL, base64 data URL, or `file://` URI with `--allowed-local-media-path`) |
| `ref_text`           | string | No       | Transcript of reference audio (for ICL mode)                                                    |
| `x_vector_only_mode` | bool   | No       | Use speaker embedding only (no ICL)                                                             |

## Streaming

Set `stream=true` with `response_format="pcm"` to receive raw PCM audio chunks as they are decoded
(one chunk per Code2Wav window, default 25 frames; configurable in the stage config):

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "vivian",
        "language": "English",
        "stream": true,
        "response_format": "pcm"
    }' --no-buffer | play -t raw -r 24000 -e signed -b 16 -c 1 -
```

**Constraints:**
- `stream=true` requires `response_format="pcm"` (raw 16-bit signed PCM, 24 kHz mono).
- `speed` adjustment is not supported when streaming.
- Requires the server stage config to have `async_chunk: true` (default in `qwen3_tts.yaml`).

## Streaming Text Input (WebSocket)

The `/v1/audio/speech/stream` WebSocket endpoint accepts text incrementally, buffers and splits it at sentence boundaries, and generates audio per sentence.

When `stream_audio=true`, each sentence is emitted as `audio.start`, one or more binary PCM frames, and `audio.done`.

### Quick Start

```bash
python streaming_speech_client.py \
    --text "Hello world. How are you? I am fine."

python streaming_speech_client.py \
    --text "Hello world. How are you? I am fine." \
    --simulate-stt --stt-delay 0.1
```

### WebSocket Protocol

Client -> Server:

```jsonc
{"type": "session.config", "voice": "Vivian", "task_type": "CustomVoice", "language": "Auto", "split_granularity": "sentence", "stream_audio": true, "response_format": "pcm"}
{"type": "input.text", "text": "Hello, how are you? "}
{"type": "input.done"}
```

Server -> Client:

```jsonc
{"type": "audio.start", "sentence_index": 0, "sentence_text": "Hello, how are you?", "format": "pcm", "sample_rate": 24000}
// binary PCM frame(s)
{"type": "audio.done", "sentence_index": 0, "total_bytes": 96000, "error": false}
{"type": "session.done", "total_sentences": 1}
```

## Limitations

- **Single request**: Batch processing is not yet optimized for online serving.

## Troubleshooting

1. **TTS model did not produce audio output**: Ensure you're using the correct model variant for your task type (CustomVoice task → CustomVoice model, etc.)
2. **Connection refused**: Make sure the server is running on the correct port
3. **Out of memory**: Use smaller model variant (`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`) or reduce `--gpu-memory-utilization`
4. **Unsupported speaker**: Use `/v1/audio/voices` to list available voices for the loaded model
5. **Voice clone fails**: Ensure you're using the Base model variant for voice cloning

## Example materials

??? abstract "gradio_demo.py"
    ``````py
    --8<-- "examples/online_serving/qwen3_tts/gradio_demo.py"
    ``````
??? abstract "gradio_fastrtc_demo.py"
    ``````py
    --8<-- "examples/online_serving/qwen3_tts/gradio_fastrtc_demo.py"
    ``````
??? abstract "openai_speech_client.py"
    ``````py
    --8<-- "examples/online_serving/qwen3_tts/openai_speech_client.py"
    ``````
??? abstract "run_gradio_demo.sh"
    ``````sh
    --8<-- "examples/online_serving/qwen3_tts/run_gradio_demo.sh"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/qwen3_tts/run_server.sh"
    ``````
??? abstract "streaming_speech_client.py"
    ``````py
    --8<-- "examples/online_serving/qwen3_tts/streaming_speech_client.py"
    ``````
??? abstract "tts_common.py"
    ``````py
    --8<-- "examples/online_serving/qwen3_tts/tts_common.py"
    ``````
