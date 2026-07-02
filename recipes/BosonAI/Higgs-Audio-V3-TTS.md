# Higgs-Audio V3 TTS

> Multilingual text-to-speech with voice cloning on 1×H100

## Summary

- Vendor: Boson AI
- Model: `bosonai/higgs-audio-v3-tts-4b`
- Task: Text-to-speech synthesis with optional voice cloning (100+ languages)
- Mode: Online serving with the OpenAI-compatible `/v1/audio/speech` API; offline batch inference
- Maintainer: @yuekaiz

## When to use this recipe

Use this recipe to serve `bosonai/higgs-audio-v3-tts-4b` for high-quality
multilingual TTS. The model generates 24 kHz speech, supports zero-shot voice
cloning from a reference clip, and handles 100+ languages with inline control
tokens for emotion, style, and prosody. The architecture is a ~4B Qwen3
backbone with fused multi-codebook embedding/head (8 codebooks × 1026 vocab,
MusicGen-style delay pattern).

## References

- Model card: [bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)
- Offline example: [`examples/offline_inference/text_to_speech/higgs_audio_v3/end2end.py`](../../examples/offline_inference/text_to_speech/higgs_audio_v3/end2end.py)
- Online example: [`examples/online_serving/text_to_speech/higgs_audio_v3/`](../../examples/online_serving/text_to_speech/higgs_audio_v3/)
- Benchmark results: see Performance section below

## Hardware Support

## GPU

### 1×H100 80GB

#### Environment

- OS: Linux
- Python: 3.12+
- CUDA: 12.x
- vLLM version: 0.22.0
- vLLM-Omni version or commit: `36e048fd` (branch `higgs-v3`)

#### Command

**Online serving:**

```bash
vllm-omni serve bosonai/higgs-audio-v3-tts-4b \
    --host 0.0.0.0 --port 8095 \
    --trust-remote-code --omni
```

The default deploy config `vllm_omni/deploy/higgs_multimodal_qwen3.yaml` is
loaded automatically by model registry (HF `model_type=higgs_multimodal_qwen3`).
It matches the high-throughput profile.

**Offline batch inference:**

```bash
python examples/offline_inference/text_to_speech/higgs_audio_v3/end2end.py \
    --texts "Hello world." "The quick brown fox jumps over the lazy dog." \
    --output-dir results/higgs_v3_wavs
```

**Offline voice clone:**

```bash
python examples/offline_inference/text_to_speech/higgs_audio_v3/end2end.py \
    --texts "Text to synthesize in the cloned voice." \
    --ref-audio path/to/reference.wav \
    --ref-text "Transcript of the reference clip." \
    --output-dir results/higgs_v3_clone
```

#### Verification

Basic TTS via curl:

```bash
curl -X POST http://localhost:8095/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "model": "bosonai/higgs-audio-v3-tts-4b",
        "input": "Hello, how are you?"
    }' --output hello.wav
```

Voice clone via Python client:

```bash
python examples/online_serving/text_to_speech/higgs_audio_v3/batch_speech_client.py \
    --base-url http://localhost:8095 \
    --model bosonai/higgs-audio-v3-tts-4b \
    --ref-audio path/to/reference.wav \
    --ref-text "Transcript of the reference." \
    --prompts "Text to clone."
```

#### Notes

- Memory usage: Stage 0 (talker, ~4B) uses ~60% GPU memory; Stage 1 (codec decoder) uses ~25%.
- Key flags: `--trust-remote-code` and `--omni` are required.
- Output: 24 kHz mono WAV.
- Voice cloning: `ref_audio` accepts WAV/FLAC/MP3; `ref_text` is optional but improves fidelity.
- Deploy config: `vllm_omni/deploy/higgs_multimodal_qwen3.yaml` (auto-discovered from `model_type`).
  - `max_num_seqs=16` for both stages.
  - Stage 0 and Stage 1 default to the same device (`0`) for single-GPU serving.
  - Stage 0 intentionally keeps `enforce_eager=true`. This preserves the Higgs-specific local MLP CUDA graph path, which is the current high-throughput default.
  - Stage 1 remains `enforce_eager=true` for the codec decoder.
- Deploy profiles:
  - High throughput: `vllm_omni/deploy/higgs_multimodal_qwen3_high_throughput.yaml`.
    Use this for medium/high concurrency. The auto-discovered
    `higgs_multimodal_qwen3.yaml` is kept as a compatibility/default alias for
    this profile.
  - Low latency: `vllm_omni/deploy/higgs_multimodal_qwen3_low_latency.yaml`.
    Use this for low-concurrency serving (for example c1-c4). It sets Stage 0
    `enforce_eager=false` and enables vLLM `FULL_DECODE_ONLY` CUDA graph through
    YAML `compilation_config`; no environment variable is required.
  - Profile details: `vllm_omni/deploy/README_higgs_audio_v3.md`.
- Performance note:
  - Do not switch the auto-discovered default Stage 0 profile to vLLM
    `FULL_DECODE_ONLY` CUDA graph without an end-to-end throughput and
    audio-quality revalidation. On the H20 SeedTTS c16/full-dataset benchmark,
    the eager Stage 0 path with Higgs local MLP CUDA graph reproduced
    ~35 audio_s/s (`1088/1088` OK per run, three runs around 134-136s). A
    separate H20 low-concurrency smoke after the FULL_DECODE audio-feedback
    capture fix showed lower c1/c4 latency, so the FULL_DECODE profile is kept
    as an explicit low-latency option rather than the throughput default.
- Known limitations:
  - Stage 1 (code2wav) must use `enforce_eager=true` (`@torch.inference_mode` incompatible with graph capture).
  - Stage 0 full-decode CUDA graph is experimental; sampler, delay-state updates, staging, and request postprocess remain outside the graph.
