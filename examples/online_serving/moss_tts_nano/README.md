# MOSS-TTS-Nano

## Model checkpoint

| Model | Description |
|-------|-------------|
| `OpenMOSS-Team/MOSS-TTS-Nano` | 0.1B AR LM + MOSS-Audio-Tokenizer-Nano codec, 48 kHz mono (mixed down from upstream stereo), ZH/EN/JA |

> **No built-in speaker presets.** Every request must include `ref_audio`.
> The server uses upstream's recommended `voice_clone` mode (per
> upstream's README and `infer.py` example). The OpenAI-schema `voice`
> and `ref_text` fields are accepted but ignored — `voice_clone` does
> not consume a transcript, and upstream's `continuation` mode (the only
> path that accepts `prompt_text`) emits near-silent output with a
> reference clip + transcript pair, so it is not exposed here.
>
> Sample reference clips are available in the upstream repo under
> [`assets/audio/`](https://github.com/OpenMOSS/MOSS-TTS-Nano/tree/main/assets/audio)
> (e.g. `zh_1.wav`, `en_2.wav`, `jp_2.wav`).

## Gradio Demo

An interactive Gradio demo is available with custom voice cloning and
streaming support. Upload your own reference audio in the UI.

```bash
# Option 1: Launch server + Gradio together
./run_gradio_demo.sh

# Option 2: If server is already running
python gradio_demo.py --api-base http://localhost:8091
```

Then open http://localhost:7860 in your browser.

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

Every request needs `ref_audio` (base64 data URL). Reuse a saved sample:

```bash
# Fetch a sample reference clip from the upstream repo (one-off).
# Cache under XDG_CACHE_HOME so it survives across runs and stays user-scoped.
REF_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/moss-tts-nano"
mkdir -p "$REF_DIR"
REF_WAV="$REF_DIR/zh_1.wav"
[ -s "$REF_WAV" ] || curl -L -o "$REF_WAV" https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS-Nano/main/assets/audio/zh_1.wav
REF_AUDIO=$(base64 -w 0 "$REF_WAV")

curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d "{
        \"input\": \"你好，这是语音合成测试。\",
        \"ref_audio\": \"data:audio/wav;base64,${REF_AUDIO}\",
        \"response_format\": \"wav\"
    }" --output output.wav
```

### Using Python

```python
import base64
import os
import urllib.request
from pathlib import Path

import httpx

ref_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "moss-tts-nano"
ref_dir.mkdir(parents=True, exist_ok=True)
ref_wav = ref_dir / "zh_1.wav"
if not ref_wav.exists() or ref_wav.stat().st_size == 0:
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS-Nano/main/assets/audio/zh_1.wav",
        ref_wav,
    )

with ref_wav.open("rb") as f:
    ref_audio_b64 = base64.b64encode(f.read()).decode("ascii")

response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "你好，这是语音合成测试。",
        "ref_audio": f"data:audio/wav;base64,{ref_audio_b64}",
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
    -d "{
        \"input\": \"Hello, streaming output from MOSS-TTS-Nano.\",
        \"ref_audio\": \"data:audio/wav;base64,${REF_AUDIO}\",
        \"stream\": true,
        \"response_format\": \"pcm\"
    }" --no-buffer | play -t raw -r 48000 -e signed -b 16 -c 1 -
```

**Note:** Output is 48 kHz mono PCM. Upstream's audio tokenizer is internally stereo at 48 kHz; the model wrapper averages the two channels into mono before reaching the engine, so playback duration / pitch are correct against the WAV header's 48 kHz rate.

## API Parameters

MOSS-TTS-Nano uses the standard `/v1/audio/speech` endpoint.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string | **required** | Text to synthesize (ZH / EN / JA) |
| `ref_audio` | string | **required** | Base64 data URL of the reference audio clip |
| `ref_text` | string | accepted, ignored | Schema-compatible field; voice_clone mode does not consume a transcript |
| `response_format` | string | `"wav"` | Audio format: wav, mp3, flac, pcm |
| `stream` | bool | false | Stream raw PCM chunks |
| `max_new_tokens` | int | 4096 | Maximum tokens to generate |

The `voice` and `ref_text` fields from the OpenAI schema are accepted but
ignored — there are no built-in speaker presets in MOSS-TTS-Nano, and
upstream's voice_clone mode does not consume a transcript.

## Troubleshooting

1. **`libnvrtc.so.13: cannot open shared object file`**: torchaudio 2.10+ defaults to torchcodec which requires NVRTC. vLLM-Omni patches this automatically at model load time to use soundfile instead.
2. **Connection refused**: Ensure the server is running on the correct port.
3. **Flashinfer version mismatch**: Set `FLASHINFER_DISABLE_VERSION_CHECK=1` if you see version warnings.
4. **Out of memory**: The default `gpu_memory_utilization=0.3` is conservative. Increase it in the stage config if you have more VRAM available.
