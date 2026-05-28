# Text-To-Speech (Online Serving)

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/text_to_speech>.


vLLM-Omni exposes TTS models through the OpenAI-compatible
[`POST /v1/audio/speech`](https://github.com/vllm-project/vllm-omni/tree/main/docs/serving/speech_api.md) endpoint,
launched with `vllm serve <model> --omni`. Each TTS model has its own
subdirectory containing client snippets, gradio demos, and helper
scripts; this README is the single doc entry point for all of them.

For offline inference, see [`examples/offline_inference/text_to_speech`](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_speech/README.md).
For the full list of supported architectures across all modalities, see
[Supported Models](https://github.com/vllm-project/vllm-omni/tree/main/docs/models/supported_models.md).

## Supported Models

| Model | HuggingFace repo | Voice cloning | Streaming | Voice presets / upload | Gradio demo |
|---|---|---|---|---|---|
| CosyVoice3 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | ✓ (`ref_audio`+`ref_text`) | ✓ (PCM stream) | — | — |
| Fish Speech S2 Pro | `fishaudio/s2-pro` | ✓ (`ref_audio`+`ref_text`) | ✓ (PCM stream) | — | ✓ |
| higgs-audio v2 | `bosonai/higgs-audio-v2-generation-3B-base` | ✓ (`ref_audio`+`ref_text`) | ✓ (codec_streaming) | — | — |
| GLM-TTS | `zai-org/GLM-TTS` | ✓ (`ref_audio`+`ref_text`, required) | ✓ (PCM stream) | — | ✓ |
| OmniVoice | `k2-fsa/OmniVoice` | (offline only) | — | — | — |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-{CustomVoice,VoiceDesign,Base}` | ✓ (Base) | ✓ (PCM + WebSocket) | ✓ (presets + `/v1/audio/voices` upload) | ✓ (standard + FastRTC) |
| VoxCPM2 | `openbmb/VoxCPM2` | ✓ | ✓ (AudioWorklet via gradio) | — | ✓ |
| Voxtral TTS | `mistralai/Voxtral-4B-TTS-2603` | ✓ (gated upstream) | ✓ | ✓ (presets) | ✓ |

For offline inference of any of these models, see the [offline TTS hub](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_speech/README.md).

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

For full request-shape documentation (all parameters, response formats, error codes), see the [Speech API reference](https://github.com/vllm-project/vllm-omni/tree/main/docs/serving/speech_api.md).

---

## CosyVoice3

2-stage TTS (`talker` + flow-matching `code2wav`) at 24 kHz. Voice cloning only — every request needs `ref_audio` + `ref_text`; there are no built-in voice presets.

### Prerequisites
```bash
huggingface-cli download FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

If your downloaded checkpoint lacks `config.json`, add one with `{"model_type": "cosyvoice3", "architectures": ["CosyVoice3Model"]}` (the loader reads `model_type` to select the class).

### Launch
```bash
vllm serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --omni --port 8091 --trust-remote-code
# or:
./cosyvoice3/run_server.sh
```

Streaming is on by default via `async_chunk: true` in `vllm_omni/deploy/cosyvoice3.yaml`. Pass `--no-async-chunk` (or `NO_ASYNC_CHUNK=1 ./cosyvoice3/run_server.sh`) for the legacy synchronous path.

### CLI client
The client defaults to the official upstream zero-shot prompt, so it runs without extra flags:
```bash
cd examples/online_serving/text_to_speech/cosyvoice3
python speech_client.py --text "收到好友从远方寄来的生日礼物。"
```

Pass your own reference clip and transcript for a different voice:
```bash
python speech_client.py --text "Hello, this is a cloned voice." \
    --ref-audio /path/to/reference.wav \
    --ref-text "Transcript of the reference audio."
```

Stream PCM instead of WAV:
```bash
python speech_client.py --text "Hello world" --stream --output output.pcm
```

The client supports `--api-base`, `--model`, `--text`, `--ref-audio`, `--ref-text`, `--response-format`, `--stream`, `--output`.

### Notes
- Stage 0 (`talker`) emits speech tokens; stage 1 (`code2wav`) runs flow matching + HiFiGAN to synthesize waveform.
- Deploy config auto-loads from `vllm_omni/deploy/cosyvoice3.yaml` based on HF `model_type`. Pass `--deploy-config <path>` to override.
- For offline inference and the end-to-end script, see the [offline CosyVoice3 section](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_speech/README.md#cosyvoice3).

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

## higgs-audio v2

2-stage TTS at 24 kHz: a vLLM-native Llama-3.2-3B talker with a DualFFN audio expert (Stage 0) feeding a HiggsAudio codec decoder (Stage 1) that streams chunks back to the client.

### Prerequisites

Voice clone uses HF's `HiggsAudioV2TokenizerModel` loaded from `k2-fsa/OmniVoice/audio_tokenizer/` (~806 MB subdir; the boson-ai standalone tokenizer Hub repo's `model.safetensors` is the 3B talker LM, not the codec):

```bash
pip install -U "transformers>=5.3.0"
```

### Launch
```bash
GPUS=6,7 PORT=8094 bash examples/online_serving/text_to_speech/higgs_audio_v2/run_server.sh
```
Deploy config auto-loads from `vllm_omni/deploy/higgs_audio_v2.yaml`.

### Sending requests
```bash
# Plain TTS
python higgs_audio_v2/batch_speech_client.py \
    --base-url http://localhost:8094 \
    --output-dir /tmp/higgs_out \
    --prompts "Hello world." "The quick brown fox jumps over the lazy dog."

# Voice cloning — pass a reference clip and its transcript together
python higgs_audio_v2/batch_speech_client.py \
    --base-url http://localhost:8094 \
    --output-dir /tmp/higgs_clone \
    --ref-audio /path/to/reference.wav \
    --ref-text  "Exact transcript spoken in reference.wav." \
    --prompts "Hello, this is a cloned voice."
```

### Notes
- Output: 24 kHz mono.
- `--ref-text` must be the real transcript of `--ref-audio`; mismatched text degrades cloned-voice quality.
- Out of scope (rejected with explicit 4xx): multi-speaker `[SPEAKERn]` tags inside `input`, `profile:` text-only speaker descriptions, the `ref_audio_in_system_message` system-block variant, chunked long-form generation, and per-request `voice` / `instructions` / `task_type` / `language` / `speed != 1.0` / `x_vector_only_mode` / `speaker_embedding`.
## GLM-TTS

2-stage TTS (AR + DiT flow-matching) at 24 kHz. Every request requires `ref_audio` + `ref_text`.

### Launch
```bash
vllm serve zai-org/GLM-TTS --omni --trust-remote-code --port 8091
# or:
bash examples/online_serving/text_to_speech/glm_tts/run_server.sh /path/to/GLM-TTS
```

### Sending requests
```bash
# Voice cloning (required)
python examples/online_serving/text_to_speech/glm_tts/openai_speech_client.py \
    --text "你好，这是语音克隆测试。" \
    --ref-audio file:///path/to/ref.wav \
    --ref-text "这是参考音频的文本内容。"

# Custom format
python examples/online_serving/text_to_speech/glm_tts/openai_speech_client.py \
    --text "Hello, this is a voice cloning test." \
    --ref-audio file:///path/to/ref.wav \
    --ref-text "Transcript of the reference audio." \
    --response-format mp3 -o output.mp3
```

### Gradio demo
```bash
bash examples/online_serving/text_to_speech/glm_tts/run_gradio_demo.sh
```

### Notes
- Output: 24 kHz mono WAV via HiFT vocoder.
- `ref_audio` + `ref_text` are **required** together on every request. Reference audio should be 3-10 seconds.
- Voice cloning feature extraction (WhisperVQ, CampPlus, mel) runs on the model side — no external dependency on the serving layer.

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
- Voice cloning and voice design require offline inference; see the [offline OmniVoice section](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_speech/README.md#omnivoice).

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

### Executor backend
Single-GPU serves now default to the uniproc executor (lower IPC overhead, the Base cloning use case from [#2603](https://github.com/vllm-project/vllm-omni/issues/2603) / [#2604](https://github.com/vllm-project/vllm-omni/pull/2604)). `vllm_omni/deploy/qwen3_tts.yaml` is the only Qwen3-TTS deploy config; pass `--deploy-config <path>` to override.

To opt out of chunked streaming, pass `--no-async-chunk` — the pipeline auto-dispatches to the end-to-end codec processor.

### Sending requests
```bash
# CustomVoice with a predefined speaker
python qwen3_tts/openai_speech_client.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --text "今天天气真好" \
    --voice ryan \
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

# FastRTC variant (gapless WebRTC streaming):
pip install fastrtc
python qwen3_tts/gradio_fastrtc_demo.py --api-base http://localhost:8000
```

### Speaker embedding interpolation
`qwen3_tts/speaker_embedding_interpolation.py` blends two predefined speakers' embeddings to produce intermediate voices. See the script for usage.

### Batch client
`qwen3_tts/batch_speech_client.py` issues many concurrent requests for throughput measurement.

### Notes
- Base voice cloning has per-request reference-audio cost; the uniproc default keeps IPC overhead off the critical path. See the executor-backend section above for background.
- `vllm_omni/deploy/qwen3_tts.yaml` is the default deploy config (loaded by HF `model_type`); per-stage runtime overrides are available via `--stage-N-<field> <value>`.

---

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

## Example materials

??? abstract "cosyvoice3/run_server.sh"

    --8<-- "examples/online_serving/text_to_speech/cosyvoice3/run_server.sh"

??? abstract "cosyvoice3/speech_client.py"

    --8<-- "examples/online_serving/text_to_speech/cosyvoice3/speech_client.py"

??? abstract "fish_speech/gradio_demo.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/fish_speech/gradio_demo.py"
    ``````
??? abstract "fish_speech/run_gradio_demo.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_speech/fish_speech/run_gradio_demo.sh"
    ``````
??? abstract "fish_speech/run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_speech/fish_speech/run_server.sh"
    ``````
??? abstract "fish_speech/speech_client.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/fish_speech/speech_client.py"
    ``````
??? abstract "higgs_audio_v2/batch_speech_client.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/higgs_audio_v2/batch_speech_client.py"
    ``````
??? abstract "higgs_audio_v2/run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_speech/higgs_audio_v2/run_server.sh"
??? abstract "glm_tts/gradio_demo.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/glm_tts/gradio_demo.py"
    ``````
??? abstract "glm_tts/openai_speech_client.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/glm_tts/openai_speech_client.py"
    ``````
??? abstract "glm_tts/run_gradio_demo.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_speech/glm_tts/run_gradio_demo.sh"
    ``````
??? abstract "glm_tts/run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_speech/glm_tts/run_server.sh"
    ``````
??? abstract "omnivoice/run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_speech/omnivoice/run_server.sh"
    ``````
??? abstract "omnivoice/speech_client.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/omnivoice/speech_client.py"
    ``````
??? abstract "qwen3_tts/batch_speech_client.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/qwen3_tts/batch_speech_client.py"
    ``````
??? abstract "qwen3_tts/gradio_demo.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/qwen3_tts/gradio_demo.py"
    ``````
??? abstract "qwen3_tts/openai_speech_client.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/qwen3_tts/openai_speech_client.py"
    ``````
??? abstract "qwen3_tts/run_gradio_demo.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_speech/qwen3_tts/run_gradio_demo.sh"
    ``````
??? abstract "qwen3_tts/run_server.sh"
    ``````sh
    --8<-- "examples/online_serving/text_to_speech/qwen3_tts/run_server.sh"
    ``````
??? abstract "qwen3_tts/speaker_embedding_interpolation.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/qwen3_tts/speaker_embedding_interpolation.py"
    ``````
??? abstract "qwen3_tts/streaming_speech_client.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/qwen3_tts/streaming_speech_client.py"
    ``````
??? abstract "qwen3_tts/tts_common.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/qwen3_tts/tts_common.py"
    ``````
??? abstract "voxcpm2/gradio_demo.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/voxcpm2/gradio_demo.py"
    ``````
??? abstract "voxcpm2/openai_speech_client.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/voxcpm2/openai_speech_client.py"
    ``````
??? abstract "voxtral_tts/gradio_demo.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/voxtral_tts/gradio_demo.py"
    ``````
??? abstract "voxtral_tts/text_preprocess.py"
    ``````py
    --8<-- "examples/online_serving/text_to_speech/voxtral_tts/text_preprocess.py"
    ``````
