# Text-To-Speech

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_speech>.


vLLM-Omni supports several autoregressive TTS models. They share a common
CLI shape (`--text`, `--ref-audio`, `--ref-text`, `--output-dir`) and live
together in this hub. Each model has its own subdirectory containing a
single `end2end.py` script; this README is the single doc entry point.

For online serving, see `examples/online_serving/<model>/`. For the full
list of supported architectures across all modalities, see
[Supported Models](https://github.com/vllm-project/vllm-omni/tree/main/docs/models/supported_models.md).

## Supported Models

| Model | HuggingFace repo | Stages | Voice cloning | Streaming | Special modes | Sample rate |
|---|---|---|---|---|---|---|
| VoxCPM2 | `openbmb/VoxCPM2` | single (native AR) | ✓ | — | continuation (`--ref-audio` + `--ref-text`) | 48 kHz |
| CosyVoice3 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | 2 (talker + code2wav) | ✓ | ✓ (`async_chunk: true` default) | — | 22.05 kHz |
| Fish Speech S2 Pro | `fishaudio/s2-pro` | dual-AR | ✓ | ✓ (`--streaming`) | — | 44.1 kHz |
| GLM-TTS | `zai-org/GLM-TTS` | 2 (AR + DiT) | ✓ (required) | ✓ | — | 24 kHz |
| OmniVoice | `k2-fsa/OmniVoice` | 2 (gen + dec) | ✓ | — | voice design (`--instruct`), language (`--lang`) | 24 kHz |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-{CustomVoice,VoiceDesign,Base}` | 2 (talker + code2wav) | ✓ (Base) | ✓ | 3 task variants (`--query-type`) | 24 kHz |
| Voxtral TTS | `mistralai/Voxtral-4B-TTS-2603` | varies | ✓ | ✓ | voice presets (`--voice`) | 24 kHz |
| higgs-audio v2 | `bosonai/higgs-audio-v2-generation-3B-base` (+ `k2-fsa/OmniVoice/audio_tokenizer/` codec) | 2 (talker + code2wav, DualFFN) | ✓ (`--ref-audio` + `--ref-text`) | — | — | 24 kHz |

## Common Quick Start

Most models share this invocation shape:

```bash
python examples/offline_inference/text_to_speech/<model>/end2end.py \
    --text "Hello, this is a test." \
    --ref-audio /path/to/reference.wav \
    --ref-text  "Transcript of the reference audio."
```

`--ref-audio` and `--ref-text` are optional (text-only synthesis works without them) and must be provided together for voice cloning. The exotic scripts — Qwen3-TTS, Voxtral TTS, CosyVoice3 — accept additional model-specific flags documented in their per-model section below.

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

### Notes
- Output: 48 kHz mono WAV.
- Stage config: `vllm_omni/model_executor/stage_configs/voxcpm2.yaml` (default).

---

## CosyVoice3

2-stage TTS pipeline (`talker` + `code2wav`) at 22.05 kHz.

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
Pass a reference audio. Note that CosyVoice3's `--prompt-text` is a system-style prompt for the GPT stage, not a reference transcript:
```bash
python examples/offline_inference/text_to_speech/cosyvoice3/end2end.py \
    --model pretrained_models/Fun-CosyVoice3-0.5B \
    --tokenizer pretrained_models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
    --ref-audio prompt.wav \
    --prompt-text "You are a helpful assistant.<|endofprompt|>Testing my voices. Why should I not?"
```

### Notes
- Stage 0 (`talker`) emits speech tokens; stage 1 (`code2wav`) runs flow matching + HiFiGAN to synthesize waveform.
- Deploy config auto-loads from `vllm_omni/deploy/cosyvoice3.yaml` based on HF `model_type`. Pass `--deploy-config <path>` to override.
- `async_chunk: true` is the default; pass `--no-async-chunk` to switch to the legacy synchronous path.

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

### Batched decoding
The Code2Wav stage supports batched decoding through the SpeechTokenizer. Configure both stages with `max_num_seqs > 1` via `--stage-overrides` and pass multiple prompts via `--txt-prompts`:
```bash
python examples/offline_inference/text_to_speech/qwen3_tts/end2end.py \
    --query-type CustomVoice \
    --txt-prompts examples/offline_inference/text_to_speech/qwen3_tts/benchmark_prompts.txt \
    --batch-size 4 \
    --stage-overrides '{"0":{"max_num_seqs":4,"gpu_memory_utilization":0.2},"1":{"max_num_seqs":4,"gpu_memory_utilization":0.2}}'
```
`--batch-size` must match a CUDA-graph capture size (1, 2, 4, 8, 16…).

### Notes
- Run `--help` for the full argument surface.
- See `qwen3_tts/end2end.py` for the prompt-length-estimation logic the Talker uses.

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

## higgs-audio v2

2-stage TTS at 24 kHz: a vLLM-native Llama-3.2-3B talker with a DualFFN audio expert (Stage 0) feeding a HiggsAudio codec decoder (Stage 1) over the shared-memory connector. Stage 1 builds on the HiggsAudio decoder kernel at `vllm_omni/model_executor/models/higgs_audio_v2/higgs_audio_decoder.py`.

### Prerequisites

Voice clone uses HF's `HiggsAudioV2TokenizerModel`, instantiated from `k2-fsa/OmniVoice/audio_tokenizer/` — the boson-ai standalone tokenizer Hub repo's `model.safetensors` is actually the 3B talker LM, so we point HF at k2's repackaged codec weights instead. Only the `audio_tokenizer/` subdir (~806 MB) is downloaded.

```bash
pip install -U "transformers>=5.3.0"
```

### Quick start (plain TTS)

```bash
python examples/offline_inference/text_to_speech/higgs_audio_v2/end2end.py \
    --texts "Hello world." "The quick brown fox jumps over the lazy dog." \
    --output-dir results/wavs
```

### Voice cloning

Pass both `--ref-audio` and `--ref-text` together:

```bash
python examples/offline_inference/text_to_speech/higgs_audio_v2/end2end.py \
    --texts "Hello, this is a cloned voice." \
    --ref-audio /path/to/reference.wav \
    --ref-text  "Exact transcript spoken in reference.wav." \
    --output-dir results/wavs
```

### Notes

- Deploy config auto-loads from `vllm_omni/deploy/higgs_audio_v2.yaml`.
- `--ref-text` must be the real transcript of `--ref-audio`; mismatched text degrades cloned-voice quality.
- Out of scope (rejected with 4xx by the request validator): multi-speaker `[SPEAKERn]` dialogue, `profile:` text-only speaker descriptions, the `ref_audio_in_system_message` system-block variant, chunked long-form generation, and per-request `voice` / `instructions` / `task_type` / `language` / `speed != 1.0` / `x_vector_only_mode` / `speaker_embedding`.

## Example materials

??? abstract "cosyvoice3/end2end.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_speech/cosyvoice3/end2end.py"
    ``````
??? abstract "fish_speech/end2end.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_speech/fish_speech/end2end.py"
    ``````
??? abstract "higgs_audio_v2/end2end.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_speech/higgs_audio_v2/end2end.py"
??? abstract "glm_tts/end2end.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_speech/glm_tts/end2end.py"
    ``````
??? abstract "omnivoice/end2end.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_speech/omnivoice/end2end.py"
    ``````
??? abstract "qwen3_tts/benchmark_prompts.txt"
    ``````txt
    --8<-- "examples/offline_inference/text_to_speech/qwen3_tts/benchmark_prompts.txt"
    ``````
??? abstract "qwen3_tts/end2end.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_speech/qwen3_tts/end2end.py"
    ``````
??? abstract "voxcpm2/end2end.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_speech/voxcpm2/end2end.py"
    ``````
??? abstract "voxtral_tts/end2end.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_speech/voxtral_tts/end2end.py"
    ``````
