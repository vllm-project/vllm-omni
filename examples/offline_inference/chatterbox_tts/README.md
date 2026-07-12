# Chatterbox TTS

Offline inference examples for [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
on vLLM-Omni. Chatterbox is a two-stage pipeline:

- **Stage 0 (T3)**: autoregressive speech-token decoder.
- **Stage 1 (S3Gen)**: CFM + HiFi-GAN vocoder producing a 24 kHz waveform.

Two variants are supported:

| Variant | Architecture | HuggingFace repo | Backbone | Status |
|---|---|---|---|---|
| Turbo | `ChatterboxTurboT3ForGeneration` | `ResembleAI/chatterbox-turbo` | GPT-2 (350M) | Production-ready |
| Original | `ChatterboxT3ForGeneration` | `ResembleAI/chatterbox` | LLaMA (520M) | Preview — quality requires AR-stage CFG (follow-up) |

For the full list of supported architectures, see
[Supported Models](../../../docs/models/supported_models.md).

## Quick Start

### Turbo — zero-shot

```bash
python examples/offline_inference/chatterbox_tts/chatterbox_tts.py \
    --text "The quick brown fox jumps over the lazy dog." \
    --output-dir /tmp/out_turbo/
```

### Turbo — voice cloning

Pass a reference audio clip to clone its speaker:

```bash
python examples/offline_inference/chatterbox_tts/chatterbox_tts.py \
    --text "The quick brown fox jumps over the lazy dog." \
    --ref-audio /path/to/reference.wav \
    --output-dir /tmp/out_turbo_clone/
```

### Original (preview)

Original generates audio end-to-end but, without classifier-free guidance at
the AR stage, output is muffled relative to native Chatterbox. The script
prints a warning to this effect; CFG support is a tracked follow-up.

```bash
python examples/offline_inference/chatterbox_tts/chatterbox_tts_original.py \
    --text "The quick brown fox jumps over the lazy dog." \
    --output-dir /tmp/out_original_preview/
```

Hardware: validated on 1× L4 (24 GB) for Turbo.
