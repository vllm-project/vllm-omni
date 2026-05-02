# MOSS-TTS-Nano

## Model checkpoint

| Model | Description |
|-------|-------------|
| `OpenMOSS-Team/MOSS-TTS-Nano` | 0.1B AR LM + MOSS-Audio-Tokenizer-Nano codec, 48 kHz stereo, ZH/EN/JA |

## Gradio Demo

An interactive Gradio demo is available with multilingual voice presets, custom voice cloning, and streaming support.

```bash
# Option 1: Launch server + Gradio together
./run_gradio_demo.sh

# Option 2: If server is already running
python gradio_demo.py --api-base http://localhost:8091
```

Then open http://localhost:7860 in your browser.

Features:

- 15 built-in voice presets (6 ZH · 4 EN · 5 JA)
- Custom voice cloning from uploaded reference audio
- Streaming mode (progressive PCM playback)

## Launch the Server

```bash
vllm serve OpenMOSS-Team/MOSS-TTS-Nano --omni --port 8091
```

The deploy config at `vllm_omni/deploy/moss_tts_nano.yaml` auto-loads; no
`--stage-configs-path`, `--trust-remote-code`, or `--enforce-eager` flags
are needed.

Or use the convenience script:

```bash
./run_server.sh
```

## Send TTS Request

### Using curl

```bash
# Built-in voice preset
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "你好，这是语音合成测试。",
        "voice": "Junhao",
        "response_format": "wav"
    }' --output output.wav

# English preset
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, this is MOSS-TTS-Nano.",
        "voice": "Ava",
        "response_format": "wav"
    }' --output output_en.wav
```

### Custom Voice Cloning

Provide a reference audio (base64 data URL) and its transcript:

```bash
REF_AUDIO=$(base64 -w 0 /path/to/reference.wav)
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d "{
        \"input\": \"Hello, this is a cloned voice.\",
        \"voice\": \"Junhao\",
        \"ref_audio\": \"data:audio/wav;base64,${REF_AUDIO}\",
        \"ref_text\": \"Exact transcript of the reference audio.\"
    }" --output cloned.wav
```

### Using Python

```python
import httpx

response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "你好，这是语音合成测试。",
        "voice": "Junhao",
        "response_format": "wav",
    },
    timeout=300.0,
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Streaming

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, streaming output from MOSS-TTS-Nano.",
        "voice": "Ava",
        "stream": true,
        "response_format": "pcm"
    }' --no-buffer | play -t raw -r 48000 -e signed -b 16 -c 2 -
```

**Note:** MOSS-TTS-Nano outputs at 48 kHz stereo (2-channel).

## API Parameters

MOSS-TTS-Nano uses the standard `/v1/audio/speech` endpoint.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | Text to synthesize (ZH / EN / JA) |
| `voice` | string | `"Junhao"` | Built-in voice name (see table below) |
| `response_format` | string | `"wav"` | Audio format: wav, mp3, flac, pcm |
| `ref_audio` | string | null | Base64 data URL for custom voice cloning |
| `ref_text` | string | null | Transcript of reference audio (required when `ref_audio` is set) |
| `stream` | bool | false | Stream raw PCM chunks |
| `max_new_tokens` | int | 4096 | Maximum tokens to generate |

## Built-in Voice Presets

| Voice | Language | Description |
|-------|----------|-------------|
| `Junhao` | ZH | Male, standard Mandarin |
| `Zhiming` | ZH | Male |
| `Weiguo` | ZH | Male |
| `Xiaoyu` | ZH | Female |
| `Yuewen` | ZH | Female |
| `Lingyu` | ZH | Female |
| `Ava` | EN | Female, American English |
| `Bella` | EN | Female |
| `Adam` | EN | Male |
| `Nathan` | EN | Male |
| `Sakura` | JA | Female |
| `Yui` | JA | Female |
| `Aoi` | JA | Female |
| `Hina` | JA | Female |
| `Mei` | JA | Female |

## Troubleshooting

1. **`libnvrtc.so.13: cannot open shared object file`**: torchaudio 2.10+ defaults to torchcodec which requires NVRTC. vLLM-Omni patches this automatically at model load time to use soundfile instead.
2. **Connection refused**: Ensure the server is running on the correct port.
3. **Flashinfer version mismatch**: Set `FLASHINFER_DISABLE_VERSION_CHECK=1` if you see version warnings.
4. **Out of memory**: The default `gpu_memory_utilization=0.3` is conservative. Increase it in the stage config if you have more VRAM available.
