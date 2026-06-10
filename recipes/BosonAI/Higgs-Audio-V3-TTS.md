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
  - `enforce_eager=true` for both stages (CUDA graph available for stage 0 but no throughput gain at batch>1).
- Known limitations:
  - CUDA graph for stage 0 helps batch=1 (-13% RTF) but not batch>1 due to model-owned sampler running outside graph.
  - Stage 1 (code2wav) must use `enforce_eager=true` (`@torch.inference_mode` incompatible with graph capture).
  - Async streaming (chunk-based) not yet implemented; sync-only pipeline.
