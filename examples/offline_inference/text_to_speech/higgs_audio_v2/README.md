# higgs-audio v2 — offline example

Drives Stage 0 (DualFFN talker) + Stage 1 (HiggsAudio codec) for `bosonai/higgs-audio-v2-generation-3B-base` end-to-end through the vLLM-Omni engine and writes a 24 kHz mono WAV per prompt.

## Prerequisites

Voice clone needs `transformers>=5.3.0` — vllm-omni loads the audio codec via HF's `HiggsAudioV2TokenizerModel`, instantiated from the `k2-fsa/OmniVoice/audio_tokenizer/` subdirectory (only that ~806 MB subdir is downloaded). The boson-ai standalone tokenizer repo's `model.safetensors` is actually a copy of the 3B talker LM, so HF can't load it directly; the k2 bundle ships the same codec weights repackaged with HF-compatible key naming.

```bash
pip install -U "transformers>=5.3.0"
```

## Quick start

Plain TTS:
```bash
python examples/offline_inference/text_to_speech/higgs_audio_v2/end2end.py \
    --texts "Hello world." "The quick brown fox jumps over the lazy dog." \
    --output-dir results/wavs
```

## Voice cloning

Pass both `--ref-audio` and `--ref-text` together:
```bash
python examples/offline_inference/text_to_speech/higgs_audio_v2/end2end.py \
    --texts "Hello, this is a cloned voice." \
    --ref-audio /path/to/reference.wav \
    --ref-text  "Exact transcript spoken in reference.wav." \
    --output-dir results/wavs
```

## Notes

- Output: 24 kHz mono WAV.
- Deploy config: `vllm_omni/deploy/higgs_audio_v2.yaml` (auto-loaded by HF `model_type`).
- `--ref-text` must be the real transcript of `--ref-audio`; mismatched text degrades cloned-voice quality.
- For online serving, see [`examples/online_serving/text_to_speech/higgs_audio_v2/`](../../../online_serving/text_to_speech/higgs_audio_v2/).
