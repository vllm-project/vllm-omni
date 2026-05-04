# Text-To-Speech (Online Serving)

vLLM-Omni exposes TTS models through the OpenAI-compatible
[`POST /v1/audio/speech`](../../../docs/serving/speech_api.md) endpoint,
launched with `vllm serve <model> --omni`. Each TTS model has its own
subdirectory containing client snippets, gradio demos, and helper
scripts; this README is the single doc entry point for all of them.

For offline inference, see [`examples/offline_inference/text_to_speech`](../../offline_inference/text_to_speech/README.md).
For the full list of supported architectures across all modalities, see
[Supported Models](../../../docs/models/supported_models.md).

## Supported Models

| Model | HuggingFace repo | Voice cloning | Streaming | Voice presets / upload | Gradio demo |
|---|---|---|---|---|---|
| Fish Speech S2 Pro | `fishaudio/s2-pro` | ✓ (`ref_audio`+`ref_text`) | ✓ (PCM stream) | — | ✓ |
| OmniVoice | `k2-fsa/OmniVoice` | (offline only) | — | — | — |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-{CustomVoice,VoiceDesign,Base}` | ✓ (Base) | ✓ (PCM + WebSocket) | ✓ (presets + `/v1/audio/voices` upload) | ✓ (standard + FastRTC) |
| VoxCPM | local model dir | ✓ | ✓ (PCM stream) | — | — |
| VoxCPM2 | `openbmb/VoxCPM2` | ✓ | ✓ (AudioWorklet via gradio) | — | ✓ |
| Voxtral TTS | `mistralai/Voxtral-4B-TTS-2603` | ✓ (gated upstream) | ✓ | ✓ (presets) | ✓ |

CosyVoice3 is intentionally absent: no online example exists for it yet. See its [offline section](../../offline_inference/text_to_speech/README.md#cosyvoice3) instead.

## Common Quick Start

Launch the server (defaults shown — adjust `--port`, `--gpu-memory-utilization`, etc. as needed):

```bash
vllm serve <hf-repo-or-local-path> --omni --port 8091
```

Send a TTS request via curl:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "default",
        "response_format": "wav"
    }' --output output.wav
```

Or via Python httpx:

```python
import httpx

response = httpx.post(
    "http://localhost:8091/v1/audio/speech",
    json={
        "input": "Hello, how are you?",
        "voice": "default",
        "response_format": "wav",
    },
    timeout=300.0,
)
open("output.wav", "wb").write(response.content)
```

Or via the OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8091/v1", api_key="none")
response = client.audio.speech.create(
    model="<hf-repo>",
    voice="default",
    input="Hello, how are you?",
)
response.stream_to_file("output.wav")
```

Streaming PCM output (where supported) — set `stream=true` with `response_format="pcm"`:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "voice": "default",
        "stream": true,
        "response_format": "pcm"
    }' --no-buffer | play -t raw -r 24000 -e signed -b 16 -c 1 -
```

Adjust the player's sample rate to match the model (44.1 kHz for Fish Speech, 48 kHz for VoxCPM2, 24 kHz for the others).

For full request-shape documentation (all parameters, response formats, error codes), see the [Speech API reference](../../../docs/serving/speech_api.md).

---

## Fish Speech S2 Pro

4B dual-AR TTS at 44.1 kHz. Server uses the DAC codec.

### Prerequisites
```bash
pip install fish-speech
```

### Launch
```bash
vllm serve fishaudio/s2-pro --omni --port 8091
# or:
./fish_speech/run_server.sh
```
The deploy config auto-loads from `vllm_omni/deploy/fish_qwen3_omni.yaml` (the HF `model_type` on the fishaudio checkpoint is `fish_qwen3_omni`).

### Voice cloning
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

### CLI client
```bash
cd examples/online_serving/text_to_speech/fish_speech
python speech_client.py --text "Hello, how are you?"
python speech_client.py --text "Hello world" --stream --output output.pcm
```

### Gradio demo
```bash
./fish_speech/run_gradio_demo.sh             # launches server + Gradio
python fish_speech/gradio_demo.py --api-base http://localhost:8091  # if server already running
```

### Notes
- Output: 44.1 kHz mono.
- Streaming PCM player command must use `-r 44100`.

---

## OmniVoice

Zero-shot multilingual TTS (600+ languages). Online serving currently exposes **auto voice** only; voice cloning and voice design are available offline.

### Prerequisites
```bash
huggingface-cli download k2-fsa/OmniVoice
```
Voice cloning (offline) needs `transformers>=5.3.0`; auto voice works with `transformers>=4.57.0`.

### Launch
```bash
vllm serve k2-fsa/OmniVoice --omni --port 8091 --trust-remote-code
# or:
./omnivoice/run_server.sh
```

### CLI client
```bash
cd examples/online_serving/text_to_speech/omnivoice
python speech_client.py --text "Hello, how are you?"
python speech_client.py --text "Bonjour, comment allez-vous?" --language French
```

The client supports `--api-base`, `--model`, `--text`, `--response-format`, `--language`, `--output`.

### Notes
- Voice cloning and voice design require offline inference; see the [offline OmniVoice section](../../offline_inference/text_to_speech/README.md#omnivoice).

---

## Qwen3-TTS

Three model variants exposed via separate checkpoints:

| Variant | HF repo | Use |
|---|---|---|
| CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Predefined speakers (`vivian`, `ryan`, …) with optional style instructions |
| VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Natural-language voice style description |
| Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Voice cloning from a reference audio |

Each variant ships smaller `0.6B` companions where available.

### Launch
```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --omni --port 8091
# or:
./qwen3_tts/run_server.sh                # default: CustomVoice
./qwen3_tts/run_server.sh VoiceDesign
./qwen3_tts/run_server.sh Base
```

### Choosing an executor backend (uniproc vs mp)
Stage configs ship with the chunked-streaming default. To use the uniproc executor (lower IPC overhead for the Base cloning task), pass `--stage-configs-path vllm_omni/model_executor/stage_configs/qwen3_tts_uniproc.yaml`. See [#2603](https://github.com/vllm-project/vllm-omni/issues/2603) and [#2604](https://github.com/vllm-project/vllm-omni/pull/2604) for the full investigation.

To opt out of chunked streaming, pass `--no-async-chunk` instead — the pipeline auto-dispatches to the end-to-end codec processor.

### Sending requests
```bash
# CustomVoice with a predefined speaker
python qwen3_tts/openai_speech_client.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --text "今天天气真好" \
    --speaker ryan \
    --instructions "用开心的语气说"

# VoiceDesign with a style description
python qwen3_tts/openai_speech_client.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --task-type VoiceDesign \
    --text "哥哥，你回来啦" \
    --instructions "体现撒娇稚嫩的萝莉女声，音调偏高"

# Base voice cloning
python qwen3_tts/openai_speech_client.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --task-type Base \
    --text "Hello, this is a cloned voice" \
    --ref-audio /path/to/reference.wav \
    --ref-text "Original transcript of the reference audio"
```

### Voices endpoint
List available voices, or upload a custom one for Base cloning:
```bash
# List
curl http://localhost:8091/v1/audio/voices

# Upload
curl -X POST http://localhost:8091/v1/audio/voices \
    -F "audio_sample=@/path/to/voice_sample.wav" \
    -F "consent=user_consent_id" \
    -F "name=custom_voice_1" \
    -F "ref_text=The exact transcript of the audio sample." \
    -F "speaker_description=warm narrator"
```
Uploaded voices are then usable as `voice="custom_voice_1"` on subsequent requests.

### Streaming PCM
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
Streaming requires `response_format="pcm"` and `async_chunk: true` on the stage config (default in `qwen3_tts.yaml`). `speed` is not supported when streaming.

### Streaming WebSocket
The `/v1/audio/speech/stream` endpoint accepts text incrementally, splits it at sentence boundaries, and emits one PCM stream per sentence:
```bash
python qwen3_tts/streaming_speech_client.py --text "Hello world. How are you? I am fine."
python qwen3_tts/streaming_speech_client.py --text "..." --simulate-stt --stt-delay 0.1
```

### Gradio demos
```bash
./qwen3_tts/run_gradio_demo.sh                              # CustomVoice (default)
./qwen3_tts/run_gradio_demo.sh --task-type VoiceDesign
./qwen3_tts/run_gradio_demo.sh --task-type Base
```

### Speaker embedding interpolation
`qwen3_tts/speaker_embedding_interpolation.py` blends two predefined speakers' embeddings to produce intermediate voices. See the script for usage.

### Batch client
`qwen3_tts/batch_speech_client.py` issues many concurrent requests for throughput measurement.

### Notes
- Base voice cloning has uniproc-vs-mp tradeoffs depending on per-request reference audio cost; see the executor-backend section above.
- `vllm_omni/deploy/qwen3_tts.yaml` is the default deploy config (loaded by HF `model_type`); per-stage runtime overrides are available via `--stage-N-<field> <value>`.

---

## VoxCPM

Split-stage TTS at 24 kHz.

### Prerequisites
```bash
pip install voxcpm
# or use a local source tree:
export VLLM_OMNI_VOXCPM_CODE_PATH=/path/to/VoxCPM/src
```

If the native VoxCPM `config.json` lacks HF `model_type`, set up an HF-compatible config dir:
```bash
export VOXCPM_MODEL=/path/to/voxcpm-model
export VLLM_OMNI_VOXCPM_HF_CONFIG_PATH=/tmp/voxcpm_hf_config
mkdir -p "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH"
cp "$VOXCPM_MODEL/config.json" "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH/config.json"
cp "$VOXCPM_MODEL/generation_config.json" "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH/generation_config.json" 2>/dev/null || true
python3 -c 'import json, os; p=os.path.join(os.environ["VLLM_OMNI_VOXCPM_HF_CONFIG_PATH"], "config.json"); cfg=json.load(open(p, "r", encoding="utf-8")); cfg["model_type"]="voxcpm"; cfg.setdefault("architectures", ["VoxCPMForConditionalGeneration"]); json.dump(cfg, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)'
```

### Launch
```bash
export VOXCPM_MODEL=/path/to/voxcpm-model
./voxcpm/run_server.sh                # async-chunk streaming (default)
./voxcpm/run_server.sh sync           # non-streaming
```
Or directly:
```bash
vllm serve "$VOXCPM_MODEL" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm_async_chunk.yaml \
    --trust-remote-code --enforce-eager --omni --port 8091
```

### Sending requests
```bash
# Basic TTS
python voxcpm/openai_speech_client.py \
    --model "$VOXCPM_MODEL" \
    --text "This is a VoxCPM online text-to-speech example."

# Voice cloning
python voxcpm/openai_speech_client.py \
    --model "$VOXCPM_MODEL" \
    --text "This sentence is synthesized with a cloned voice." \
    --ref-audio /path/to/reference.wav \
    --ref-text "The exact transcript spoken in reference.wav."

# Streaming PCM
python voxcpm/openai_speech_client.py \
    --model "$VOXCPM_MODEL" \
    --text "This is a streaming VoxCPM request." \
    --stream --output voxcpm_stream.pcm
```

### Notes
- `voxcpm.yaml` for one-shot decode; `voxcpm_async_chunk.yaml` for single-request streaming. Do not use the async-chunk config for concurrent requests or `/v1/audio/speech/batch`.
- Generic TTS fields not supported by VoxCPM: `voice`, `instructions`, `language`, `speaker_embedding`, `x_vector_only_mode`.
- For benchmark measurement, see [`benchmarks/voxcpm`](../../../benchmarks/voxcpm/README.md).

---

## VoxCPM2

Single-stage native AR TTS at 48 kHz.

### Launch
```bash
vllm serve openbmb/VoxCPM2 --omni --host 0.0.0.0 --port 8000
```
Deploy config auto-loads from `vllm_omni/deploy/voxcpm2.yaml`. Pass `--deploy-config <path>` to override or `--stage-N-<field> <value>` for per-stage runtime tweaks.

### Sending requests
```bash
# Zero-shot synthesis
python voxcpm2/openai_speech_client.py --text "Hello, this is VoxCPM2."

# Voice cloning
python voxcpm2/openai_speech_client.py \
    --text "This should sound like the reference speaker." \
    --ref-audio /path/to/reference.wav
```
The `ref_audio` field accepts local file paths (auto-base64), HTTP URLs, or `data:audio/wav;base64,...` data URIs.

### Gradio demo (gapless streaming via AudioWorklet)
```bash
python voxcpm2/gradio_demo.py
```
Uses an AudioWorklet-based player adapted from the Qwen3-TTS demo for gap-free playback. Audio is streamed from the OpenAI Speech endpoint with `stream=true`.

---

## Voxtral TTS

Voxtral-4B-TTS (Mistral). Uses the `mistral_common` `SpeechRequest` protocol; voice presets are model-specific.

### Prerequisites
Latest `mistral_common` with `SpeechRequest` support:
```bash
pip install -e /path/to/mistral-common  # or upgrade from PyPI when available
```

### Launch
```bash
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni --port 8091
```
Deploy config auto-loads from `vllm_omni/deploy/voxtral_tts.yaml`.

### Gradio demo
```bash
python voxtral_tts/gradio_demo.py
```
The demo handles voice-preset selection and reference-audio upload. `voxtral_tts/text_preprocess.py` provides the text-normalization helpers used by the demo (also available for other clients).

### Notes
- Voice presets are listed on the HF model card (`mistralai/Voxtral-4B-TTS-2603`).
- Voice cloning is gated upstream and may require a recent `mistral_common`.
- A standalone CLI client is not yet shipped; the gradio demo is the canonical reference for now.
