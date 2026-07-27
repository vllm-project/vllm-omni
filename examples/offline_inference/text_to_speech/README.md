# Text-To-Speech

vLLM-Omni supports several autoregressive TTS models. They share a mostly
common CLI shape (`--text`, `--ref-audio`, `--ref-text`, plus an
output-path flag — `--output-dir` for most, `--output` for OmniVoice) and
live together in this hub. Each model has its own subdirectory containing
a single `end2end.py` script; this README is the single doc entry point.

For online serving, see [`examples/online_serving/text_to_speech/`](../../online_serving/text_to_speech/README.md). For the full
list of supported architectures across all modalities, see
[Supported Models](../../../docs/models/supported_models.md).

## Supported Models

| Model | HuggingFace repo | Stages | Voice cloning | Streaming | Special modes | Sample rate |
|---|---|---|---|---|---|---|
| CosyVoice3 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | 2 (talker + code2wav) | ✓ | ✓ | — | 24 kHz |
| Fish Speech S2 Pro | `fishaudio/s2-pro` | dual-AR | ✓ | ✓ | — | 44.1 kHz |
| GLM-TTS | `zai-org/GLM-TTS` | 2 (AR + DiT) | ✓ (required) | ✓ | — | 24 kHz |
| Ming-omni-tts | `inclusionAI/Ming-omni-tts-0.5B` | 2 (AR + audio VAE) | ✓ | ✓ | style / IP / dialect / TTA / podcast | 44.1 kHz |
| Ming-flash-omni-TTS | `Jonathan1909/Ming-flash-omni-2.0` | single (talker only) | — (caption-controlled) | — | style / IP / basic captions | 44.1 kHz |
| MOSS-TTS-Nano | `OpenMOSS-Team/MOSS-TTS-Nano` | single (AR + codec) | ✓ (required) | ✓ | voice_clone, continuation | 48 kHz |
| OmniVoice | `k2-fsa/OmniVoice` | 2 (gen + dec) | ✓ | — | voice design, language hint | 24 kHz |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-{CustomVoice,VoiceDesign,Base}` | 2 (talker + code2wav) | ✓ (Base) | ✓ | 3 task variants | 24 kHz |
| VoxCPM2 | `openbmb/VoxCPM2` | single (native AR) | ✓ | ✓ (online) | continuation | 48 kHz |
| IndexTTS-2 | `IndexTeam/IndexTTS-2` | 2 (AR talker + S2Mel DiT + BigVGAN) | ✓ (required) | — | emotion control (`--emo-audio`, `--emo-text`, `--emo-vector`) | 22.05 kHz |
| Voxtral TTS | `mistralai/Voxtral-4B-TTS-2603` | varies | ✓ | ✓ | voice presets | 24 kHz |

## Common Quick Start

Most models share this invocation shape:

```bash
python examples/offline_inference/text_to_speech/<model>/end2end.py \
    --text "Hello, this is a test." \
    --ref-audio /path/to/reference.wav \
    --ref-text  "Transcript of the reference audio."
```

`--ref-audio` and `--ref-text` are optional (text-only synthesis works without them) and must be provided together for voice cloning. The exotic scripts — Qwen3-TTS, Voxtral TTS, CosyVoice3 — accept additional model-specific flags documented in their per-model section below. Qwen3-TTS in particular uses its own argparse surface (`--query-type`, `--audio-path`, etc.) and does not follow the common shape; see its section.

---

## CosyVoice3

2-stage TTS pipeline (`talker` + `code2wav`) at 24 kHz.

### Prerequisites
```bash
uv pip install -e .
# Includes soundfile, onnxruntime, x-transformers, einops via requirements.
```

Download the model snapshot:
```python
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

If your downloaded checkpoint lacks `config.json`, add it:
```json
{
    "model_type": "cosyvoice3",
    "architectures": ["CosyVoice3Model"]
}
```
This is required because `AutoConfig.register("cosyvoice3", CosyVoice3Config)` only registers the class mapping; the loader still reads `model_type` from `config.json` to select the class.

### Quick start
```bash
python examples/offline_inference/text_to_speech/cosyvoice3/end2end.py \
    --model pretrained_models/Fun-CosyVoice3-0.5B \
    --tokenizer pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN
```

### Voice cloning
If `--ref-audio` is omitted, the script downloads the upstream
[`zero_shot_prompt.wav`](https://github.com/FunAudioLLM/CosyVoice/blob/main/asset/zero_shot_prompt.wav) from the CosyVoice repo into the current directory.
To use your own clip, pass `--ref-audio /path/to/reference.wav`, and modify `--prompt-text` correspondingly.
```bash
python examples/offline_inference/text_to_speech/cosyvoice3/end2end.py \
    --model pretrained_models/Fun-CosyVoice3-0.5B \
    --tokenizer pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
    --ref-audio /path/to/reference.wav \
    --prompt-text "You are a helpful assistant.<|endofprompt|>Trascript in your ref audio clip"
```

### Streaming
Streaming is enabled by default via `async_chunk: true` in `vllm_omni/deploy/cosyvoice3.yaml`. Pass `--no-async-chunk` on `vllm serve` to switch to the legacy synchronous path.

### Notes
- Stage 0 (`talker`) emits speech tokens; stage 1 (`code2wav`) runs flow matching + HiFiGAN to synthesize waveform.
- Deploy config auto-loads from `vllm_omni/deploy/cosyvoice3.yaml` based on HF `model_type`. Pass `--deploy-config <path>` to override.

---

## GLM-TTS

2-stage TTS pipeline (AR + DiT flow-matching) at 24 kHz. Every request requires reference audio and its transcript for zero-shot voice cloning.

### Quick start
```bash
python examples/offline_inference/text_to_speech/glm_tts/end2end.py \
    --model zai-org/GLM-TTS \
    --text "你好，这是语音合成测试。" \
    --ref-audio /path/to/reference.wav \
    --ref-text "这是参考音频的文本内容。" \
    --output-dir ./output
```

### Architecture
```
Text → [Stage 0: AR] → Speech Tokens → [Stage 1: DiT + HiFT] → Audio (24 kHz)
        (Llama-based)    (32k vocab)      (Flow Matching)
```

### Notes
- `--ref-audio` and `--ref-text` are **required** together; GLM-TTS does not support text-only synthesis.
- Reference audio should be 3-10 seconds.
- First run may be slow due to lazy loading of WhisperVQ tokenizer and CampPlus ONNX speaker embedder.
- Default sampling: temperature=1.0, top_k=25, top_p=0.8 (RAS method).
- The `--model` path should point to the repository root (not `llm/` subdirectory).

---

## Fish Speech S2 Pro

4B dual-AR text-to-speech model from FishAudio with the DAC codec at 44.1 kHz.

### Prerequisites
```bash
pip install fish-speech
```

### Quick start
```bash
python examples/offline_inference/text_to_speech/fish_speech/end2end.py \
    --text "Hello, this is a test of the Fish Speech text to speech system."
```

### Voice cloning
```bash
python examples/offline_inference/text_to_speech/fish_speech/end2end.py \
    --text "Hello, this is a cloned voice." \
    --ref-audio /path/to/reference.wav \
    --ref-text  "Transcript of the reference audio."
```

### Streaming
```bash
python examples/offline_inference/text_to_speech/fish_speech/end2end.py \
    --text "Hello, this is a streaming test." \
    --streaming
```
Streaming requires `async_chunk: true` in the stage config.

### Notes
- Output: 44.1 kHz mono WAV.
- DAC codec weights (`codec.pth`) are loaded lazily from the model directory.

---

## Ming-omni-tts

Dense 0.5B two-stage TTS pipeline (`AR + flow` + audio VAE) at 44.1 kHz. The example covers style, IP voice, music-only generation, text-to-audio events, emotion, dialect, zero-shot cloning, podcast, speech+BGM, and speech+environment-sound cases.

### Quick start
```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case style \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

### Voice cloning
```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case zero_shot \
    --ref-audio /path/to/reference.wav \
    --ref-text "在此奉劝大家别乱打美白针。" \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

### Streaming
```bash
python examples/offline_inference/text_to_speech/ming_tts/end2end.py \
    --case basic \
    --ref-audio /path/to/reference.wav \
    --streaming \
    --deploy-config vllm_omni/deploy/ming_tts.yaml \
    --enforce-eager
```

### Notes
- `style`, `ip`, `bgm`, and `tta` do not require reference audio.
- Reference-audio cases use `--ref-audio`; `zero_shot` also requires `--ref-text`.
- `podcast` uses multiple references via `--ref-audio-paths`.
- Full case details live in [`ming_tts/README.md`](ming_tts/README.md).

---

## Ming-flash-omni-TTS

Standalone talker-only deployment of Ming-flash-omni-2.0 at 44.1 kHz. Voice is controlled through caption fields (`风格` / `IP` / `语速`/`基频`/`音量`) rather than reference audio.

### Prerequisites
The example calls into `vllm_omni.model_executor.models.ming_flash_omni.prompt_utils` for the default prompt and instruction builder; no extra pip install on top of the base vLLM-Omni install.

### Quick start
```bash
python examples/offline_inference/text_to_speech/ming_flash_omni_tts/end2end.py --case style
```

### Cases
```bash
# ASMR-style whisper (caption-driven)
python examples/offline_inference/text_to_speech/ming_flash_omni_tts/end2end.py --case style

# IP voice (preset character voice via caption)
python examples/offline_inference/text_to_speech/ming_flash_omni_tts/end2end.py --case ip

# Basic speed/pitch/volume control
python examples/offline_inference/text_to_speech/ming_flash_omni_tts/end2end.py --case basic
```

Override the default text per case with `--text`, write to a custom path with `--output`.

### Notes
- Talker-only deployment — for the multimodal Ming-flash-omni example, see [`examples/offline_inference/ming_flash_omni/`](../../ming_flash_omni/).
- Deploy config: `vllm_omni/deploy/ming_flash_omni_tts.yaml` (single GPU, `enforce_eager`, `max_num_seqs: 1`).
- Decode defaults from the Ming cookbook: `max_decode_steps=200`, `cfg=2.0`, `sigma=0.25`, `temperature=0.0`, `use_zero_spk_emb=True`.

---

## MOSS-TTS-Nano

Single-stage 0.1B AR LM + MOSS-Audio-Tokenizer-Nano codec at 48 kHz mono (mixed down from upstream stereo). ZH / EN / JA. Every request requires a reference clip via `--ref-audio`.

> **No built-in speaker presets.** `--ref-audio` is required on every call. Default `--mode voice_clone` matches upstream's recommended workflow; `--mode continuation` is exposed for completeness but upstream's continuation-with-prompt path emits very short / near-silent output, so it is rarely useful in practice. Sample reference clips ship in the upstream repo under [`assets/audio/`](https://github.com/OpenMOSS/MOSS-TTS-Nano/tree/main/assets/audio) (e.g. `zh_1.wav`, `en_2.wav`, `jp_2.wav`).

### Quick start
```bash
# Fetch a sample reference clip (one-off, user-scoped cache).
REF_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/moss-tts-nano"
mkdir -p "$REF_DIR"
[ -s "$REF_DIR/zh_1.wav" ] || \
    curl -L -o "$REF_DIR/zh_1.wav" https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS-Nano/main/assets/audio/zh_1.wav

python examples/offline_inference/text_to_speech/moss_tts_nano/end2end.py \
    --text "你好，这是MOSS-TTS-Nano的语音合成演示。" \
    --ref-audio "$REF_DIR/zh_1.wav"
```
The first run downloads `OpenMOSS-Team/MOSS-TTS-Nano` and `OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano` from Hugging Face.

### Reproducible runs
```bash
python examples/offline_inference/text_to_speech/moss_tts_nano/end2end.py \
    --text "Deterministic test." \
    --ref-audio "$REF_DIR/en_2.wav" \
    --seed 42
```

### Notes
- Output: 48 kHz mono WAV (the tokenizer is internally stereo at 48 kHz; the wrapper averages to mono before reaching the engine).
- Deploy config: `vllm_omni/deploy/moss_tts_nano.yaml` (auto-loaded; override with `--deploy-config`).
- Default `--max-new-frames 375` ≈ 14 s of audio; raise for longer outputs.
- `--ref-text` is rejected in `voice_clone` mode and required only with `--mode continuation`.
- Run `--help` for the full sampling-knob surface (`--audio-temperature`, `--audio-top-k`, `--audio-top-p`, `--text-temperature`).

---

## OmniVoice

Zero-shot multilingual TTS supporting 600+ languages, with three modes (auto / clone / design).

### Prerequisites
```bash
huggingface-cli download k2-fsa/OmniVoice
```
Voice cloning requires `transformers>=5.3.0`. Auto and design modes work with `transformers>=4.57.0`.

### Quick start (auto voice)
```bash
python examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "Hello, this is a test."
```

### Voice cloning
```bash
python examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "Hello, this is a test." \
    --ref-audio ref.wav \
    --ref-text  "This is the reference transcription."
```

### Voice design
```bash
python examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "Hello, this is a test." \
    --instruct "female, low pitch, british accent"
```

### Language hint
```bash
python examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "你好，这是一个测试。" \
    --lang zh
```

### Seed for Reproducibility
```bash
python examples/offline_inference/text_to_speech/omnivoice/end2end.py \
    --model k2-fsa/OmniVoice \
    --text "Hello, this is a test." \
    --seed 42
```

### Notes
- Stage 0 (Generator): Qwen3-0.6B with 32-step iterative unmasking.
- Stage 1 (Decoder): HiggsAudioV2 RVQ + DAC at 24 kHz.

---

## Qwen3-TTS

3-task-variant TTS with 24 kHz output. Has its own argparse surface (this script does not follow the common `--text` / `--ref-audio` shape).

### Prerequisites
For ROCm builds, replace `onnxruntime` with `onnxruntime-rocm`:
```bash
pip uninstall onnxruntime
pip install onnxruntime-rocm
```

### Task variants
- `CustomVoice`: predefined speaker (speaker ID) with optional style instruction.
- `VoiceDesign`: text + descriptive instruction designs a new voice.
- `Base`: voice cloning from reference audio + transcript.

```bash
# Single sample
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type CustomVoice
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type VoiceDesign
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type Base

# Base with a custom reference audio (Qwen3-TTS uses --audio-path, not --ref-audio):
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
    --query-type Base --audio-path /path/to/reference.wav

# Base variant has an additional mode flag:
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type Base --mode-tag icl       # default
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type Base --mode-tag xvec_only # x_vector_only_mode

# Batch (multiple prompts in one run)
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py --query-type CustomVoice --use-batch-sample
```

### Streaming
```bash
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
    --query-type CustomVoice \
    --streaming \
    --output-dir /tmp/out_stream
```
Streaming requires `async_chunk: true` in the stage config.

### Word Timestamps
Generate a WAV offline and a JSON sidecar with word-level timestamps from
`Qwen/Qwen3-ForcedAligner-0.6B`:
```bash
python examples/offline_inference/text_to_speech/qwen3_tts/word_timestamps.py \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --forced-aligner Qwen/Qwen3-ForcedAligner-0.6B \
    --text "Hello world." \
    --output-dir /tmp/qwen3_tts_timestamps
```
The script writes `qwen3_tts_word_timestamps.wav` and
`qwen3_tts_word_timestamps.json`. On machines without a local CUDA toolkit,
set `VLLM_USE_FLASHINFER_SAMPLER=0` to avoid FlashInfer sampler JIT.

### Batched decoding
The Code2Wav stage supports batched decoding through the SpeechTokenizer. Pass multiple prompts via `--txt-prompts` and set `--batch-size` accordingly. To raise `max_num_seqs` on either stage, point `--stage-configs-path` at a stage configs YAML with the desired values (see `vllm_omni/model_executor/stage_configs/` for templates):
```bash
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
    --query-type CustomVoice \
    --txt-prompts examples/offline_inference/text_to_speech/qwen3_tts/benchmark_prompts.txt \
    --batch-size 4 \
    --stage-configs-path /path/to/qwen3_tts_batched.yaml
```
`--batch-size` must match a CUDA-graph capture size (1, 2, 4, 8, 16…).

### Notes
- Run `--help` for the full argument surface.
- See `qwen3_tts/end2end.py` for the prompt-length-estimation logic the Talker uses.

---

## VoxCPM2

Single-stage native AR TTS at 48 kHz. Pipeline: `feat_encoder → MiniCPM4 → FSQ → residual_lm → LocDiT → AudioVAE`.

### Prerequisites
```bash
pip install voxcpm
# or, for a local source checkout:
export VLLM_OMNI_VOXCPM_CODE_PATH=/path/to/voxcpm
```

### Quick start
```bash
python examples/offline_inference/text_to_speech/voxcpm2/end2end.py \
    --model openbmb/VoxCPM2 \
    --text "Hello, this is a VoxCPM2 demo."
```

### Voice cloning
Pass a reference audio for isolated cloning, or both `--ref-audio` + `--ref-text` for prompt continuation:
```bash
python examples/offline_inference/text_to_speech/voxcpm2/end2end.py \
    --text "Hello, this is a voice clone demo." \
    --ref-audio /path/to/reference.wav \
    --ref-text  "Transcript of the reference audio."
```

### Streaming
Streaming is exposed through the online OpenAI Speech API (`stream=true`). See [`examples/online_serving/text_to_speech/voxcpm2/gradio_demo.py`](../../online_serving/text_to_speech/voxcpm2/gradio_demo.py) for an AudioWorklet-based gapless streaming player; the offline `end2end.py` script does not expose a streaming path.

### Notes
- Output: 48 kHz mono WAV.
- Deploy config: `vllm_omni/deploy/voxcpm2.yaml` (auto-loaded by HF `model_type`).

---

## IndexTTS-2

2-stage TTS pipeline (GPT AR talker + S2Mel CFM DiT + BigVGAN vocoder) at 22.05 kHz. Every request requires reference audio for zero-shot voice cloning. Supports emotion conditioning via audio, text, or 8-dim vector.

### Quick start
```bash
python examples/offline_inference/text_to_speech/indextts2/end2end.py \
    --model IndexTeam/IndexTTS-2 \
    --text "你好，这是一个语音合成测试。" \
    --ref-audio /path/to/reference.wav
```

### Emotion control
```bash
# Emotion from reference audio
python examples/offline_inference/text_to_speech/indextts2/end2end.py \
    --model IndexTeam/IndexTTS-2 \
    --text "今天天气真好！" \
    --ref-audio /path/to/ref.wav \
    --emo-audio /path/to/happy.wav

# Emotion from 8-dim vector (happy angry sad afraid disgusted melancholy surprised calm)
python examples/offline_inference/text_to_speech/indextts2/end2end.py \
    --model IndexTeam/IndexTTS-2 \
    --text "今天天气真好！" \
    --ref-audio /path/to/ref.wav \
    --emo-vector 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0

# Emotion from text description
python examples/offline_inference/text_to_speech/indextts2/end2end.py \
    --model IndexTeam/IndexTTS-2 \
    --text "今天天气真好！" \
    --ref-audio /path/to/ref.wav \
    --emo-text "happy and excited"
```

### Notes
- `--ref-audio` is **required** — IndexTTS-2 does not support text-only synthesis.
- Stage 0 (AR Talker): GPT-2 generates mel codes from text + reference audio.
- Stage 1 (S2Mel + BigVGAN): CFM DiT converts mel codes to waveform at 22.05 kHz.
- Deploy config: `vllm_omni/deploy/indextts2.yaml`. Stage 1 runs with `enforce_eager: true` (DiT has dynamic shapes).

---

## Voxtral TTS

Voxtral-4B-TTS (Mistral). Has its own argparse surface; uses voice presets and the `mistral_common` `SpeechRequest` protocol.

### Prerequisites
Latest `mistral_common` with `SpeechRequest` support:
```bash
pip install -e /path/to/mistral-common  # or upgrade from PyPI when available
```

### Quick start (voice preset)
```bash
python examples/offline_inference/text_to_speech/voxtral_tts/end2end.py \
    --write-audio --voice cheerful_female \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "That eerie silence after the first storm was just the calm before another round of chaos, wasn't it?"
```

### Voice cloning (capability gated upstream)
```bash
python examples/offline_inference/text_to_speech/voxtral_tts/end2end.py \
    --write-audio \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "This is a test message." \
    --ref-audio path/to/reference_audio.wav
```

### Streaming + concurrency
```bash
python examples/offline_inference/text_to_speech/voxtral_tts/end2end.py \
    --num-prompts 32 --concurrency 8 --streaming --write-audio --voice neutral_female \
    --model mistralai/Voxtral-4B-TTS-2603 \
    --text "..."
```
Available voice presets are listed on the HF model card (`mistralai/Voxtral-4B-TTS-2603`).

### Notes
- `--num-prompts N` replicates the prompt for performance measurement.
- `--concurrency M` requires `--streaming` and must evenly divide `--num-prompts`.
- Run `--help` for the full argument surface.

---

## SoulX-Singer

Singing voice synthesis (SVS) and conversion (SVC) at 24 kHz. Script: `soulxsinger/end2end.py`. Deploy: `vllm_omni/deploy/soulxsinger_svs.yaml` or `soulxsinger_svc.yaml`.

### Prerequisites

Download DiT and preprocess weights, then set up separate SVS / SVC view directories. Copy `soulxsinger/utils/phoneme/phone_set.json` from upstream [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) into the model weights dir as `phoneme/phone_set.json` — HuggingFace does not ship it.

```bash
# 1. DiT weights
export BASE=path/to/SoulX-Singer
export PREPROCESS=path/to/SoulX-Singer-Preprocess
export SVC_DIR=path/to/SoulX-Singer-svc

huggingface-cli download Soul-AILab/SoulX-Singer --local-dir "$BASE"

# 2. Preprocess weights (required)
huggingface-cli download Soul-AILab/SoulX-Singer-Preprocess --local-dir "$PREPROCESS"
export SOULX_PREPROCESS_WEIGHTS_DIR="$PREPROCESS"

# 3. SVS / SVC view directories
mkdir -p "$SVC_DIR"
cp $BASE/{config.yaml,README.md,assets} $SVC_DIR
mv $BASE/model-svc.pt $SVC_DIR/model-svc.pt

cat > "$BASE/config.json" <<'EOF'
{
  "model_type": "soulxsinger",
  "architectures": ["SoulXSingerPipeline"],
  "max_num_seqs": 1
}
EOF

cat > "$SVC_DIR/config.json" <<'EOF'
{
  "model_type": "soulxsinger",
  "architectures": ["SoulXSingerSVCPipeline"],
  "max_num_seqs": 1
}
EOF
```

`config.yaml` hyper-parameters live under `$BASE`; each view's `config.json` `architectures` field is the single source of truth for SVS vs SVC. Point `--model` at the matching directory (`$BASE` for SVS, `$SVC_DIR` for SVC). Deploy YAML is chosen automatically from `config.json`; optional `--svs` / `--svc` only assert the mode matches.

**Online preprocess** is the default: pass `--prompt-audio` and `--target-audio`, and the worker runs vocal separation, F0, and (for SVS) lyrics/MIDI before DiT. Install only what your run needs:

```bash
pip install "BS-RoFormer"   # vocal sep + F0 on GPU — SVS and SVC
```

Mandarin SVS also needs FunASR and Chinese G2P; `ffmpeg` must be on `PATH`:

```bash
# install optional dependencies:
pip install -e ".[soulx-svs]"
```

English SVS adds NeMo ASR and NLTK data; pass `--language English`:

```bash
pip install "nemo_toolkit[asr]==2.6.1" lhotse==1.32.2
python -c "import nltk; nltk.download('cmudict'); nltk.download('averaged_perceptron_tagger_eng')"
```

**Precomputed metadata** is the alternative: pass both `--prompt-metadata-path` and `--target-metadata-path` and skip online ASR/ROSVOT — none of the packages above are required. JSON can be produced by integrated preprocess on a prior run, or by upstream [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) `preprocess/` scripts if you prefer to run that outside vLLM-Omni.

### Quick start

```bash
# SVS — default demo audio: tests/assets/soulxsinger/zh_prompt.mp3 + music.mp3
python examples/offline_inference/text_to_speech/soulxsinger/end2end.py \
    --model "$BASE" \
    --preprocess-weights-dir "$PREPROCESS" \
    --control score \
    --num-inference-steps 32 \
    -o output.wav

python examples/offline_inference/text_to_speech/soulxsinger/end2end.py \
    --model "$SVC_DIR" \
    --preprocess-weights-dir "$PREPROCESS" \
    --svc \
    --num-inference-steps 32 \
    -o output_svc.wav
```

`SOULX_PREPROCESS_WEIGHTS_DIR` makes `--preprocess-weights-dir` optional. Long SVS targets are handled in one request. See `end2end.py --help` for `--pitch-shift`, `--vocal-sep`, `--auto-shift`, and language/control options.

### Notes

- Output: 24 kHz mono WAV; batch only.
- Defaults match upstream: `--guidance-scale 3.0`, `--seed 42`, `--auto-shift` on.
- SVS `--control`: `score` or `melody`. MIDI / lyric QC: upstream `midi_editor` only.

---
