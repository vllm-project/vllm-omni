# higgs-audio v2 offline example

This directory contains the higgs-audio v2 offline inference scaffolding for vllm-omni:

- `reference_hf.py` — runs the upstream HF model (`bosonai/higgs-audio-v2-generation-3B-base` + `bosonai/higgs-audio-v2-tokenizer`) on a pinned prompt with greedy decode and saves the per-prompt fixture set (`tests/fixtures/higgs_audio_v2/reference_*.pt`) plus a human-readable upstream trace memo (`vllm_omni/model_executor/models/higgs_audio_v2/UPSTREAM_TRACE.md`).
- `end2end.py` — exercises the vllm-omni higgs_audio_v2 path in two modes:
  - `--mode hf_reference` runs the upstream HF reference and saves a 24 kHz WAV (downloads the boson-ai checkpoints if not cached).
  - `--mode stage1_only` replays a saved fixture's `[8, T]` code tensor through vllm-omni's Stage-1 decoder (`HiggsAudioV2Code2Wav`) to validate AC-4 (Stage-1 decode parity) without invoking the 3B Stage-0 talker.

## Prerequisites

- A CUDA-capable GPU. The defaults target H100/A100 80GB (see DEC-3 in `results/plan.md`).
- A vllm-omni environment with `transformers >= 5.3.0` and the boson-ai checkpoints accessible via HF cache.
- A project-local `.venv`:

  ```bash
  source /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/tts/vllm-omni/.venv/bin/activate
  ```

## Capture the reference fixtures (run once)

```bash
CUDA_VISIBLE_DEVICES=6,7 \
HF_HOME=/path/to/hf-cache \
python examples/offline_inference/text_to_speech/higgs_audio_v2/reference_hf.py \
    --prompts "Hello world." \
    --max-new-tokens 50 \
    --output-dir tests/fixtures/higgs_audio_v2 \
    --write-trace
```

Outputs:
- `tests/fixtures/higgs_audio_v2/reference_hello_world.pt`
- `vllm_omni/model_executor/models/higgs_audio_v2/UPSTREAM_TRACE.md`

## Validate Stage-1 decode parity

```bash
python examples/offline_inference/text_to_speech/higgs_audio_v2/end2end.py \
    --mode stage1_only \
    --fixture tests/fixtures/higgs_audio_v2/reference_hello_world.pt \
    --audio-tokenizer-dir ~/.cache/huggingface/hub/models--bosonai--higgs-audio-v2-tokenizer/snapshots/<rev>/audio_tokenizer \
    --output-wav stage1_replay.wav \
    --compare-with-reference
```

The `--compare-with-reference` flag prints the normalized-float PCM RMS between the vllm-omni Stage-1 output and the upstream HF decode; AC-4 requires RMS <= 1e-4.

## Run the HF reference end-to-end

```bash
CUDA_VISIBLE_DEVICES=6,7 \
python examples/offline_inference/text_to_speech/higgs_audio_v2/end2end.py \
    --mode hf_reference \
    --text "Hello world." \
    --output-wav hello_hf_reference.wav
```

This is useful as a smoke test when validating a fresh install before exercising the vllm-omni Stage-0 talker integration. The 3B Stage-0 talker is structurally registered in this repo but its AR-loop integration is gated on the upstream-trace memo and reference fixtures captured by `reference_hf.py`; see the model package `__init__.py` and `vllm_omni/model_executor/models/higgs_audio_v2/UPSTREAM_TRACE.md` for the contract.

## Scope (v1)

Plain text -> 24 kHz speech only. Voice cloning, multi-speaker dialogue, ChatML rich content, and reference audio are rejected with explicit 4xx by the request validator in `vllm_omni/entrypoints/openai/serving_speech.py`. See `results/plan.md` AC-5 for the full rejection list.
