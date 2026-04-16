# Fish Speech S2 Pro

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/fish_speech>.


## Model

| Model | Description |
|-------|-------------|
| `fishaudio/s2-pro` | Fish Speech S2 Pro -- 4B dual-AR TTS model with DAC codec (44.1 kHz) |

## Gradio Demo

An interactive Gradio demo is available with text-to-speech synthesis, voice cloning, and streaming support.

```bash
# Option 1: Launch server + Gradio together
./run_gradio_demo.sh

# Option 2: If server is already running
python gradio_demo.py --api-base http://localhost:8091
```

Then open http://localhost:7860 in your browser.

Features:

- Text-to-speech synthesis
- Voice cloning from uploaded audio or URL
- Streaming mode (progressive PCM playback)

## Launch the Server

```bash
vllm-omni serve fishaudio/s2-pro \
    --stage-configs-path vllm_omni/model_executor/stage_configs/fish_speech_s2_pro.yaml \
    --omni \
    --port 8091 \
    --trust-remote-code \
    --enforce-eager \
    --gpu-memory-utilization 0.9
```

Or use the convenience script:

```bash
./run_server.sh
```

## Send TTS Request

### Using curl

```bash
# Basic TTS
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "default",
        "response_format": "wav"
    }' --output output.wav
```

### Voice Cloning

Provide a reference audio (URL or base64 data URL) and its transcript:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is a cloned voice.",
        "voice": "default",
        "ref_audio": "https://example.com/reference.wav",
        "ref_text": "Transcript of the reference audio."
    }' --output cloned.wav
```

### Using Python

```python
import httpx

# Basic TTS
response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "Hello, how are you?",
        "voice": "default",
        "response_format": "wav",
    },
    timeout=300.0,
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Using the CLI Client

```bash
cd examples/online_serving/fish_speech

# Basic TTS
python speech_client.py --text "Hello, how are you?"

# Voice cloning
python speech_client.py \
    --text "Hello, this is a cloned voice." \
    --ref-audio /path/to/reference.wav \
    --ref-text "Transcript of the reference audio."

# Streaming PCM output
python speech_client.py --text "Hello world" --stream --output output.pcm
```

The CLI client supports:

- `--api-base`: API base URL (default: `http://localhost:8091`)
- `--model` (or `-m`): Model name (default: `fishaudio/s2-pro`)
- `--text`: Text to synthesize (required)
- `--ref-audio`: Reference audio for voice cloning (local path or URL)
- `--ref-text`: Transcript of the reference audio
- `--stream`: Enable streaming (PCM output)
- `--response-format`: Audio format: wav, mp3, flac, pcm, aac, opus (default: wav)
- `--output` (or `-o`): Output file path

## Streaming

Set `stream=true` with `response_format="pcm"` to receive raw PCM audio chunks as they are decoded:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "default",
        "stream": true,
        "response_format": "pcm"
    }' --no-buffer | play -t raw -r 44100 -e signed -b 16 -c 1 -
```

**Note:** Fish Speech outputs at 44.1 kHz (unlike Qwen3-TTS which outputs at 24 kHz).

## API Parameters

Fish Speech uses the same `/v1/audio/speech` endpoint as Qwen3-TTS. See the [Speech API reference](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/) for full parameter documentation.

Key parameters for Fish Speech:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | Text to synthesize |
| `voice` | string | "default" | Voice name (use "default" for Fish Speech) |
| `response_format` | string | "wav" | Audio format: wav, mp3, flac, pcm, aac, opus |
| `ref_audio` | string | null | Reference audio URL or base64 data URL for voice cloning |
| `ref_text` | string | null | Transcript of reference audio (required for voice cloning) |
| `max_new_tokens` | int | 4096 | Maximum tokens to generate |
| `stream` | bool | false | Stream raw PCM chunks |

## Prerequisites

Install the `fish-speech` package for the DAC codec:

```bash
pip install fish-speech
```

## Troubleshooting

1. **No audio output**: Make sure the `fish-speech` package is installed for the DAC decoder
2. **Connection refused**: Ensure the server is running on the correct port
3. **Flashinfer version mismatch**: Set `FLASHINFER_DISABLE_VERSION_CHECK=1` if you see version warnings
4. **Out of memory**: Reduce `--gpu-memory-utilization` or use a GPU with more VRAM

## Example materials

??? abstract "gradio_demo.py"
    ``````py
    --8<-- "examples/online_serving/fish_speech/gradio_demo.py"
    ``````
??? abstract "run_gradio_demo.sh"
    ``````sh
    --8<-- "examples/online_serving/fish_speech/run_gradio_demo.sh"
    ``````
??? abstract "run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/fish_speech/run_server.sh"
    ``````
??? abstract "speech_client.py"
    ``````py
    --8<-- "examples/online_serving/fish_speech/speech_client.py"
    ``````
