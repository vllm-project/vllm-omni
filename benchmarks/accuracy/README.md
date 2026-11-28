# Accuracy Benchmarks

This directory hosts accuracy benchmark integrations that run entirely through a
local `vllm-omni serve` deployment.

Current integrations:

- `text_to_image/`: GEBench generation + local judge scoring flow.
- `image_to_image/`: GEdit-Bench generation + local VIEScore-style scoring flow.
- `text_to_speech/`: Seed-TTS generation + WER/SIM/UTMOS evaluation.

Design notes:

- Generation is executed through the OpenAI-compatible endpoints exposed by
  `vllm-omni serve`.
- Evaluation is also executed through a local OpenAI-compatible judge model
  served by `vllm-omni`.
- Both generation and judge requests accept either `http://host:port` or
  `http://host:port/v1`.
- Output directory layout intentionally stays close to the upstream repos.

## Text-to-Speech (Seed-TTS)

Standalone script: `text_to_speech/seed_tts_bench.py`

Usage:
```bash
# Prerequisites
pip install 'vllm-omni[dev]'  # For WER/SIM/UTMOS evaluation

# Start vLLM server
vllm serve Qwen/Qwen3-TTS --omni --port 8000

# Run evaluation (English)
python benchmarks/accuracy/text_to_speech/seed_tts_bench.py \
    --model Qwen/Qwen3-TTS \
    --locale en \
    --output-dir ./results/tts-en

# Run evaluation (Mandarin)
python benchmarks/accuracy/text_to_speech/seed_tts_bench.py \
    --model Qwen/Qwen3-TTS \
    --locale zh \
    --output-dir ./results/tts-zh
```

Metrics (same as `vllm bench serve --omni --wer-eval`):
- **WER** (0-1): Word Error Rate from speech-to-text (lower = better)
  - English: OpenAI Whisper-large-v3
  - Mandarin: Alibaba Paraformer-zh
- **SIM** (0-1): Speaker similarity via WavLM embeddings (higher = better)
- **UTMOS** (0-5): Mean Opinion Score prediction (higher = better)

Output: JSON with per-prompt and aggregate metrics at `{output_dir}/summary_*.json`

## Inline vs. Standalone

**Standalone Scripts** (this directory):
- ✓ Focused accuracy evaluation
- ✓ Easy to script and automate
- ✓ Self-contained, no benchmark overhead

**Inline Evaluation** (`vllm bench serve --omni`):
- ✓ Performance + quality metrics together
- ✓ Concurrent request testing
- ✓ Streaming metrics (if supported)

Both use identical evaluation logic from `vllm_omni/benchmarks/data_modules/seed_tts_eval.py`,
so metrics are directly comparable.

Test guidance:

- Local static/self-checks live in `tests/benchmarks/test_accuracy_bench_utils.py`.
- Standalone accuracy scripts also include consistency tests (e.g., `text_to_speech/test_seed_tts_consistency.py`).
- End-to-end generation/evaluation should be validated in a remote GPU
  environment. In the current repo marker system there is `L4` but no `L5`
  marker, so benchmark smoke tests should be wired as `full_model +
  benchmark + L4` for nightly when GPU capacity is available.
