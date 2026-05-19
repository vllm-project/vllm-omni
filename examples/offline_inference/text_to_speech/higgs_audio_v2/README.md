# higgs-audio v2 offline tools

This directory contains offline helpers for the vllm-omni `higgs_audio_v2` model.

- `recog_wav.py` — ASR-based sanity check: loads an English offline transducer
  (sherpa-onnx-nemo-fast-conformer) and transcribes one or more synthesized
  WAV files, optionally comparing against an expected prompt with a simple
  word-level WER.

The online-serving entry points live under
`examples/online_serving/text_to_speech/higgs_audio_v2/` (Gradio demo,
batch speech client, `run_server.sh`). The Stage-0 talker + Stage-1 codec
implementation is under
`vllm_omni/model_executor/models/higgs_audio_v2/`; the upstream architecture
contract is documented in `UPSTREAM_TRACE.md` in that directory.

## Usage

```bash
python examples/offline_inference/text_to_speech/higgs_audio_v2/recog_wav.py \
    --wav /tmp/hello_world.wav \
          /tmp/the_quick_brown_fox.wav \
    --expected "Hello world." \
               "The quick brown fox jumps over the lazy dog."
```

## Scope (v1)

Plain text -> 24 kHz speech only. Voice cloning, multi-speaker dialogue,
ChatML rich content, language overrides, `task_type`, `speed != 1.0`, and
reference audio are rejected with explicit 4xx by the request validator in
`vllm_omni/entrypoints/openai/serving_speech.py`.
