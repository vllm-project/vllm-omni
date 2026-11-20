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
vllm serve bosonai/higgs-audio-v3-tts-4b \
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
- Reference substitution diagnostics: temporarily set `VLLM_LOGGING_LEVEL=DEBUG`
  before starting the server to see why reference audio substitution was skipped.
  Count-mismatch messages report the batch-local request index, placeholder count,
  and reference code row count without logging text or audio content. They do not
  change the existing fallback behavior; legacy counts describe the current span.
- Deploy config: `vllm_omni/deploy/higgs_multimodal_qwen3.yaml` (auto-discovered from `model_type`).
    - Uses `model_runner: v2`; vLLM 0.34 no longer supports V1.
    - MRV2 switches self-enable from the runner selection; no enabling `hf_overrides` block is needed on CUDA.
    - Both stages use `max_num_seqs=16` and device `0` by default.
    - CUDA Stage 0 uses FLASH_ATTN, `FULL` CUDA graph capture and `async_scheduling: true`; Stage 1 stays eager with codec graph windows `[8, 33]`.
    - CUDA streaming uses 25-frame chunks with 8-frame left context and an 8-frame first chunk (~320 ms of audio).
    - Non-CUDA overrides preserve eager execution and the earlier streaming
      geometry, use V2 with async scheduling, and disable the six CUDA-tuned
      `audio_mrv2_*` optimizations. ROCm, NPU, XPU, and MUSA are not supported
      on this vLLM 0.34 head; these overrides are retained only for
      configuration/test coverage until platform-specific quality and
      performance validation is available.
    - The sampler supports `temperature`, `top_p`, `top_k` and `seed`. Remove unsupported penalties, logprobs, min_tokens, min_p, allowed/bad token ids and logit_bias; they are rejected before queue admission.
- Deploy profiles:
    - Default: `vllm_omni/deploy/higgs_multimodal_qwen3.yaml`.
    - H200 tuning: `vllm_omni/deploy/higgs_multimodal_qwen3_mrv2_h200.yaml` (64-sequence capacities, larger capture sizes, MPS).
    - The old `_high_throughput.yaml` and `_low_latency.yaml` files retain V1 execution settings and are unavailable as V1 opt-outs with vLLM 0.34.
    - Profile details: `vllm_omni/deploy/README_higgs_audio_v3.md`.
- Validation notes:
    - Earlier V1/V2 quality and performance comparisons predate the vLLM 0.34 rebase. They do not qualify the current runtime; repeat matched hardware measurements before making new parity or speedup claims.
    - Stage 1 must use `enforce_eager=true`; its codec graphs are managed locally.
