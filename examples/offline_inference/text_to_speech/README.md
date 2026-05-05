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
| CosyVoice3 | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | 2 (talker + code2wav) | ✓ | ✓ | — | 22.05 kHz |
| Fish Speech S2 Pro | `fishaudio/s2-pro` | dual-AR | ✓ | ✓ | — | 44.1 kHz |
| OmniVoice | `k2-fsa/OmniVoice` | 2 (gen + dec) | ✓ | — | voice design, language hint | 24 kHz |
| Qwen3-TTS | `Qwen/Qwen3-TTS-12Hz-1.7B-{CustomVoice,VoiceDesign,Base}` | 2 (talker + code2wav) | ✓ (Base) | ✓ | 3 task variants | 24 kHz |
| VoxCPM | local model dir | split | ✓ | ✓ | — | 24 kHz |
| VoxCPM2 | `openbmb/VoxCPM2` | single (native AR) | ✓ | ✓ (online) | continuation | 48 kHz |
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

### Streaming
Streaming is enabled by default via `async_chunk: true` in `vllm_omni/deploy/cosyvoice3.yaml`. Pass `--no-async-chunk` on `vllm serve` to switch to the legacy synchronous path.

### Notes
- Stage 0 (`talker`) emits speech tokens; stage 1 (`code2wav`) runs flow matching + HiFiGAN to synthesize waveform.
- Deploy config auto-loads from `vllm_omni/deploy/cosyvoice3.yaml` based on HF `model_type`. Pass `--deploy-config <path>` to override.

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

## VoxCPM

Split-stage TTS. The hub example covers single TTS, single voice cloning, and streaming. Use `benchmarks/voxcpm/` for warmup, batch JSONL prompts, profiler injection, and offline TTFP / RTF measurement.

### Prerequisites
```bash
pip install voxcpm soundfile
# or use a local source tree:
export VLLM_OMNI_VOXCPM_CODE_PATH=/path/to/VoxCPM/src
```

If the native VoxCPM `config.json` does not contain HF metadata such as `model_type`, prepare a persistent HF-compatible config directory and point the stage configs to it via `VLLM_OMNI_VOXCPM_HF_CONFIG_PATH`:

```bash
export VOXCPM_MODEL=/path/to/voxcpm-model
export VLLM_OMNI_VOXCPM_HF_CONFIG_PATH=/tmp/voxcpm_hf_config
mkdir -p "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH"
cp "$VOXCPM_MODEL/config.json" "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH/config.json"
cp "$VOXCPM_MODEL/generation_config.json" "$VLLM_OMNI_VOXCPM_HF_CONFIG_PATH/generation_config.json" 2>/dev/null || true
python3 -c 'import json, os; p=os.path.join(os.environ["VLLM_OMNI_VOXCPM_HF_CONFIG_PATH"], "config.json"); cfg=json.load(open(p, "r", encoding="utf-8")); cfg["model_type"]="voxcpm"; cfg.setdefault("architectures", ["VoxCPMForConditionalGeneration"]); json.dump(cfg, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)'
```

### Quick start
```bash
python examples/offline_inference/text_to_speech/voxcpm/end2end.py \
    --model "$VOXCPM_MODEL" \
    --text "This is a split-stage VoxCPM synthesis example running on vLLM Omni."
```

### Voice cloning
```bash
python examples/offline_inference/text_to_speech/voxcpm/end2end.py \
    --model "$VOXCPM_MODEL" \
    --text "This sentence is synthesized with a cloned voice." \
    --ref-audio /path/to/reference.wav \
    --ref-text  "The exact transcript spoken in reference.wav."
```

### Streaming
Pass the async-chunk stage config:
```bash
python examples/offline_inference/text_to_speech/voxcpm/end2end.py \
    --model "$VOXCPM_MODEL" \
    --stage-configs-path vllm_omni/model_executor/stage_configs/voxcpm_async_chunk.yaml \
    --text "This is a split-stage VoxCPM streaming example running on vLLM Omni."
```

### Notes
- `voxcpm.yaml` is the default non-streaming stage config; `voxcpm_async_chunk.yaml` enables streaming.
- Streaming is currently single-request oriented.
- `--ref-text` must be the real transcript of `--ref-audio`; mismatched text degrades quality.
- For online serving, see the [VoxCPM section in the online hub](../../online_serving/text_to_speech/README.md#voxcpm). For benchmark reporting, see [`benchmarks/voxcpm`](../../../benchmarks/voxcpm/README.md).

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
