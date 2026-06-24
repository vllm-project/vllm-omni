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
| GLM-TTS | `zai-org/GLM-TTS` | ✓ (`ref_audio`+`ref_text`, required) | ✓ (PCM stream) | — | ✓ |
| IndexTTS-2 | `IndexTeam/IndexTTS-2` | ✓ (`ref_audio` or uploaded `voice`) | compat only, non-chunk | uploaded audio voice only; no presets | — |
| Ming-omni-tts | `inclusionAI/Ming-omni-tts-0.5B` | ✓ (`ref_audio` / `speaker_embedding`) | ✓ (PCM stream) | IP labels + structured `instructions` | — |
| Ming-flash-omni-TTS | `Jonathan1909/Ming-flash-omni-2.0` | — (caption-controlled) | — | caption fields (`instructions`) | — |
| MOSS-TTS-Nano | `OpenMOSS-Team/MOSS-TTS-Nano` | ✓ (`ref_audio` required) | ✓ (PCM stream) | — | ✓ |
| OmniVoice | `k2-fsa/OmniVoice` | ✓ | — | — | — |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-{CustomVoice,VoiceDesign,Base}` | ✓ (Base) | ✓ (PCM + WebSocket) | ✓ (presets + `/v1/audio/voices` upload) | ✓ (standard + FastRTC) |
| VoxCPM2 | `openbmb/VoxCPM2` | ✓ | ✓ (AudioWorklet via gradio) | — | ✓ |
| Voxtral TTS | `mistralai/Voxtral-4B-TTS-2603` | ✓ (gated upstream) | ✓ | ✓ (presets) | ✓ |
| SoulX-Singer | `Soul-AILab/SoulX-Singer` | ✓ (prompt audio) | — (batch only) | — (prompt + target audio) | — (chat client) |

CosyVoice3 is intentionally absent: no online example exists for it yet. See its [offline section](../../offline_inference/text_to_speech/README.md#cosyvoice3) instead.

## Common Quick Start

Launch the server (defaults shown — adjust `--port`, `--gpu-memory-utilization`, etc. as needed):

```bash
vllm serve <hf-repo-or-local-path> --omni --port 8091
```

Send a TTS request via curl. These generic snippets assume a model with a preset/default voice; voice-cloning-only models such as IndexTTS-2 require `ref_audio` or an uploaded audio `voice` (see model-specific sections below).

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

Adjust the player's sample rate to match the model (44.1 kHz for Fish Speech, 48 kHz for VoxCPM2, 22.05 kHz for IndexTTS-2, and 24 kHz for many others).

For full request-shape documentation (all parameters, response formats, error codes), see the [Speech API reference](../../../docs/serving/speech_api.md).

---

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

## IndexTTS-2

2-stage TTS (GPT AR + S2Mel CFM DiT + BigVGAN) at 22.05 kHz. Requests use `ref_audio` for voice cloning, or an uploaded audio `voice` from `/v1/audio/voices`. Supports emotion conditioning via `emo_audio`, `emo_text`, or `emo_vector` passed in `extra_params`.

### Launch
```bash
vllm serve IndexTeam/IndexTTS-2 --omni --trust-remote-code --port 8092
# or, to pass the bundled deploy config explicitly:
bash examples/online_serving/text_to_speech/indextts2/run_server.sh
```

### Sending requests
```bash
# Voice cloning (ref_audio required)
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
    --text "你好，世界！" \
    --ref-audio /path/to/reference.wav

# With emotion audio
python examples/online_serving/text_to_speech/indextts2/speech_client.py \
    --text "今天心情很好！" \
    --ref-audio /path/to/ref.wav \
    --emo-audio /path/to/happy.wav
```

### Notes
- Output: 22.05 kHz mono WAV.
- Provide `ref_audio` on the documented raw request path, or pass `voice` only when it names an uploaded audio voice; IndexTTS-2 does not provide a built-in text-only preset voice.
- Emotion params (`emo_audio`, `emo_text`, `emo_vector`, `emo_alpha`, `use_emo_text`, `use_random`) are passed via the `extra_params` field. Official precedence is `use_emo_text` > `emo_vector` > `emo_audio` > same emotion as the speaker reference.
- `stream=true` is accepted as an OpenAI-compatible response path, but IndexTTS-2 is not async-chunk streaming; audio is produced after S2Mel receives the full mel-code sequence.
- Deploy config: `vllm_omni/deploy/indextts2.yaml` (auto-loaded).

---

## Fish Speech S2 Pro

4B dual-AR TTS at 44.1 kHz. Server uses the DAC codec.

### Prerequisites
```bash
pip install fish-speech
```

### Kvcache attention fast path

Fish Speech S2 Pro uses a Triton decode-only kvcache attention fast path by
default on CUDA builds. Set `VLLM_OMNI_FISH_KVCACHE_ATTN=0` to disable it, or
`VLLM_OMNI_FISH_KVCACHE_ATTN=required` to fail fast if the fast path cannot be
installed.

```bash
# Verify fast path availability.
python - <<'PY'
from vllm_omni.attention import fish_kvcache_attn

print(fish_kvcache_attn.is_available())
print(fish_kvcache_attn.load_error())
PY

# Optional: disable the runtime fast path.
export VLLM_OMNI_FISH_KVCACHE_ATTN=0
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

## Ming-omni-tts

Dense 0.5B two-stage TTS served through `/v1/audio/speech`. Ming uses the standard speech endpoint plus structured controls in `instructions`, `voice`, `language`, `ref_audio`, `ref_text`, and `speaker_embedding`.

### Launch
```bash
bash examples/online_serving/text_to_speech/ming_tts/run_server.sh
```
Equivalent manual command:
```bash
vllm-omni serve inclusionAI/Ming-omni-tts-0.5B \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --host 0.0.0.0 --port 8091 \
    --enforce-eager --omni
```

### Sending requests
```bash
python examples/online_serving/text_to_speech/ming_tts/openai_speech_client.py \
    --text "你好，这是 Ming 在线语音合成测试。"
```

Structured dialect control:
```bash
python examples/online_serving/text_to_speech/ming_tts/openai_speech_client.py \
    --text "我觉得社会企业同个人都有责任" \
    --instruction-json '{"方言":"广粤话"}' \
    --ref-audio /path/to/yue_prompt.wav
```

Zero-shot cloning:
```bash
python examples/online_serving/text_to_speech/ming_tts/openai_speech_client.py \
    --text "我们的愿景是构建未来服务业的数字化基础设施，为世界带来更多微小而美好的改变。" \
    --ref-audio /path/to/10002287-00000094.wav \
    --ref-text "在此奉劝大家别乱打美白针。"
```

### Notes
- `run_curl.sh` keeps a small sanity subset; use the Ming README for the broader request cookbook.
- Online serving is speech-shaped today; music-only `bgm` and text-to-audio `tta` remain offline examples.
- Full request details live in [`ming_tts/README.md`](ming_tts/README.md).

---

## Ming-flash-omni-TTS

Standalone talker-only deployment of Ming-flash-omni-2.0. Voice is controlled through caption text passed via `instructions`.

### Launch
```bash
# from repo root
bash examples/online_serving/text_to_speech/ming_flash_omni_tts/run_server.sh
```
Equivalent manual command:
```bash
vllm serve Jonathan1909/Ming-flash-omni-2.0 \
    --deploy-config vllm_omni/deploy/ming_flash_omni_tts.yaml \
    --host 0.0.0.0 --port 8091 \
    --trust-remote-code --omni
```

### Sending requests
```bash
python examples/online_serving/text_to_speech/ming_flash_omni_tts/speech_client.py \
    --text "我们当迎着阳光辛勤耕作，去摘取，去制作，去品尝，去馈赠。" \
    --output ming_online.wav
```

ASMR-style caption via `instructions`:
```bash
python examples/online_serving/text_to_speech/ming_flash_omni_tts/speech_client.py \
    --text "我会一直在这里陪着你，直到你慢慢、慢慢地沉入那个最温柔的梦里……好吗？" \
    --instructions "这是一种ASMR耳语，属于一种旨在引发特殊感官体验的创意风格。这个女性使用轻柔的普通话进行耳语，声音气音成分重。" \
    --output ming_online_asmr.wav
```

### Notes
- Server uses `use_zero_spk_emb=True` and the cookbook decode defaults (`max_decode_steps=200`, `cfg=2.0`, `sigma=0.25`, `temperature=0.0`). For other caption fields (`语速`, `基频`, `IP`, BGM, etc.) or overriding decode args, use the offline example where `additional_information` is set explicitly.
- This is the online counterpart of [`examples/offline_inference/text_to_speech/ming_flash_omni_tts/`](../../offline_inference/text_to_speech/ming_flash_omni_tts/).
- For multimodal Ming-flash-omni online serving, see [`examples/online_serving/ming_flash_omni/`](../../ming_flash_omni/).

---

## MOSS-TTS-Nano

Single-stage 0.1B AR LM + MOSS-Audio-Tokenizer-Nano codec at 48 kHz mono. Every request must include `ref_audio`; there are no built-in speaker presets.

> The OpenAI-schema `voice` and `ref_text` fields are accepted but ignored — `voice_clone` does not consume a transcript, and upstream's `continuation` mode (the only path that accepts `prompt_text`) emits near-silent output, so it is not exposed here. Sample reference clips ship in the upstream repo under [`assets/audio/`](https://github.com/OpenMOSS/MOSS-TTS-Nano/tree/main/assets/audio).

### Launch
```bash
vllm serve OpenMOSS-Team/MOSS-TTS-Nano --omni --port 8091
# or:
./moss_tts_nano/run_server.sh
```
The deploy config at `vllm_omni/deploy/moss_tts_nano.yaml` auto-loads; no `--stage-configs-path`, `--trust-remote-code`, or `--enforce-eager` flags are needed.

### Sending requests
```bash
# One-off fetch of a sample reference clip; cache under XDG_CACHE_HOME.
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

### Streaming PCM
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

### Gradio demo
```bash
# Option 1: launch server + Gradio together
./moss_tts_nano/run_gradio_demo.sh

# Option 2: server already running
python moss_tts_nano/gradio_demo.py --api-base http://localhost:8091
```
Then open http://localhost:7860 in your browser.

### Notes
- Output is 48 kHz mono PCM (the upstream tokenizer is internally stereo at 48 kHz; the wrapper averages to mono before reaching the engine).
- Standard `/v1/audio/speech` request shape: `input`, `ref_audio` (base64 data URL), `response_format`, `stream`, `max_new_tokens`. The `voice` and `ref_text` fields from the OpenAI schema are accepted but ignored.

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
# Text-only (auto voice)
python speech_client.py --text "Hello, how are you?"

# Language hint
python speech_client.py --text "Bonjour, comment allez-vous?" --language French
# Voice cloning (reference audio + optional ref_text)
python speech_client.py \
--text "Bonjour, comment allez-vous?" \
--ref-audio /path/to/ref_audio.wav \
--ref-text "Bonjour, comment allez-vous?"

# Style instruction (voice design-style control)
python speech_client.py \
--text "Bonjour, comment allez-vous?" \
--language French \
--instructions "loud voice"

# Deterministic output with seed parameter
python speech_client.py --text "Hello, how are you?" --seed 42
```

The client supports `--api-base`, `--model`, `--text`, `--response-format`, `--language`, `--voice`, `--ref-audio`, `--ref-text`, `--instructions`, `--seed`, and `--output`.


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

### Precomputed custom voices
For reused Base voice-cloning speakers, precompute the reference artifacts once and load them at server startup:
```bash
python qwen3_tts/precompute_custom_voice.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --voice-name alice \
    --ref-audio /path/to/reference.wav \
    --ref-text "Original transcript of the reference audio" \
    --mode icl \
    --output-dir /path/to/custom_voices
```
`--mode icl` stores both `speaker_embedding` and `ref_code`; `--mode xvec` stores only the speaker embedding. Add the output directory to a deploy config:
```yaml
custom_voice_dir: /path/to/custom_voices
```
Then start the server with that config and call the Speech API with only the voice name:
```bash
vllm serve Qwen/Qwen3-TTS-12Hz-1.7B-Base --omni --deploy-config /path/to/qwen3_tts_custom_voice.yaml

curl -X POST http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input":"Hello from a precomputed voice.","voice":"alice","task_type":"Base"}' \
    --output alice.wav
```

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

To receive word-level timestamps, launch the server with a forced aligner:
```bash
vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --omni \
    --deploy-config vllm_omni/deploy/qwen3_tts.yaml \
    --trust-remote-code \
    --forced-aligner Qwen/Qwen3-ForcedAligner-0.6B
```
Then request PCM JSON sidecar chunks:
```bash
python qwen3_tts/streaming_speech_client.py \
    --text "Hello world. How are you?" \
    --stream-audio \
    --response-format pcm \
    --word-timestamps
```
The client writes one PCM file per sentence and a matching
`sentence_XXX_timestamps.json` sidecar.

To *see* the alignment instead of reading a JSON sidecar, run the
word-timestamp Gradio demo (server must be launched with `--forced-aligner`):
```bash
python qwen3_tts/word_timestamps_demo.py --api-base http://localhost:8091
```
Each sentence's audio plays in an `<audio>` element while its text is rendered
as inline word spans; the current word highlights as `audio.currentTime`
crosses each `start_ms`. The **Stop (barge-in)** button cuts playback and
reports the last-spoken word, useful for the voice-agent barge-in case.

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
- With async chunking, Qwen3-TTS Base voice cloning sends the full reference context in the first Code2Wav packet, then caches that prefix on the Code2Wav stage for follow-up chunks in the same request.
- `vllm_omni/deploy/qwen3_tts.yaml` is the default deploy config (loaded by HF `model_type`); per-stage runtime overrides are available via `--stage-N-<field> <value>`.

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

### Precomputed custom voices
For repeated VoxCPM2 speakers, precompute the prompt cache and load it through `custom_voice_dir`:
```bash
python voxcpm2/precompute_custom_voice.py \
    --model openbmb/VoxCPM2 \
    --voice-name alice \
    --ref-audio /path/to/reference.wav \
    --mode ref_continuation \
    --prompt-text "Original transcript of the reference audio" \
    --output-dir /path/to/custom_voices
```
Add the output directory to the deploy config:
```yaml
custom_voice_dir: /path/to/custom_voices
```
After startup, `/v1/audio/voices` lists `alice`, and `/v1/audio/speech` can use `voice="alice"` without sending `ref_audio`.

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

---

## SoulX-Singer

Singing voice synthesis (SVS) and conversion (SVC) at 24 kHz. Single-stage DiT with inline preprocess. Uses the `/v1/chat/completions` endpoint with multimodal input (`prompt_audio` + `target_audio`).

### Prerequisites

Download DiT and preprocess weights, then set up separate SVS / SVC view directories and install dependencies as described in the [offline README](../../offline_inference/text_to_speech/README.md#soulx-singer). `config.json` `architectures` field is the single source of truth for SVS vs SVC — point `MODEL` at the matching directory.

### Launch

```bash
# SVS (default)
export MODEL=/path/to/SoulX-Singer
export PREPROCESS=/path/to/SoulX-Singer-Preprocess
bash examples/online_serving/text_to_speech/soulxsinger/run_server.sh

# SVC
export MODE=svc
export MODEL=/path/to/SoulX-Singer-svc
bash examples/online_serving/text_to_speech/soulxsinger/run_server.sh
```

Or equivalently, set `SOULX_PREPROCESS_WEIGHTS_DIR` and launch directly:
```bash
export SOULX_PREPROCESS_WEIGHTS_DIR=$PREPROCESS
vllm serve $MODEL --omni \
    --deploy-config vllm_omni/deploy/soulxsinger_${MODE}.yaml \
    --port 8192 --trust-remote-code --enforce-eager
```

### Sending requests

Audio paths must be reachable from the server host (local filesystem or data URL). The client sends prompt vocal via `input_audio` and target accompaniment via `extra_args['target_audio']`.

```bash
# Default demo audio: tests/assets/soulxsinger/zh_prompt.mp3 + music.mp3
python examples/online_serving/text_to_speech/soulxsinger/openai_chat_client.py \
    --prompt-audio /path/on/server/zh_prompt.mp3 \
    --target-audio /path/on/server/music.mp3 \
    --preprocess-weights-dir /path/on/server/SoulX-Singer-Preprocess \
    -o output.wav
```

Use precomputed metadata to skip online preprocess with following command:
```bash
python examples/online_serving/text_to_speech/soulxsinger/openai_chat_client.py \
    --prompt-metadata-path /path/on/server/zh_prompt.json \
    --target-metadata-path /path/on/server/music.json \
    --audio-path /path/on/server/zh_prompt.mp3 \
    -o output.wav
```

`SOULX_PREPROCESS_WEIGHTS_DIR` makes `--preprocess-weights-dir` optional. See `openai_chat_client.py --help` for `--vocal-sep`, `--language`, `--num-inference-steps`, `--guidance-scale`, and `--seed`.

### Notes

- Output: 24 kHz mono WAV; batch only.
- Defaults match upstream: `--guidance-scale 3.0`, `--seed 42`, `--auto-shift` on.
- SVS `--control`: `score` or `melody`. MIDI / lyric QC: upstream `midi_editor` only.
