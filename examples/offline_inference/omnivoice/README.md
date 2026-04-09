# OmniVoice

This directory contains an offline demo for running OmniVoice TTS models with vLLM Omni. It generates speech from text and saves WAV files locally.

## Model Overview

[OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) is a zero-shot multilingual TTS model supporting 600+ languages. It uses a diffusion language model (Qwen3-0.6B backbone) with iterative masked unmasking to generate speech.

Three inference modes are supported:

- **Auto Voice**: Generate speech without any reference — the model picks a voice automatically.
- **Voice Clone**: Clone a voice from a reference audio + transcription.
- **Voice Design**: Control voice style via natural language instruction (e.g., "female, low pitch, british accent").

## Setup

Ensure the model is downloaded:

```bash
huggingface-cli download k2-fsa/OmniVoice
```

> **Note:** Voice cloning requires `transformers>=5.3.0` for `HiggsAudioV2TokenizerModel`. Auto voice and voice design modes work with `transformers>=4.57.0`.

## Quick Start

Auto voice (text only):

```bash
python end2end.py --model k2-fsa/OmniVoice --text "Hello, this is a test."
```

Voice design (with style instruction):

```bash
python end2end.py --model k2-fsa/OmniVoice \
    --text "Hello, this is a test." \
    --instruct "female, low pitch, british accent"
```

Voice clone (with reference audio):

```bash
python end2end.py --model k2-fsa/OmniVoice \
    --text "Hello, this is a test." \
    --ref-audio ref.wav \
    --ref-text "This is the reference transcription."
```

## Language Support

Specify a language for improved quality:

```bash
python end2end.py --model k2-fsa/OmniVoice \
    --text "你好，这是一个测试。" \
    --lang zh
```

## Architecture

OmniVoice uses a two-stage pipeline:

- **Stage 0 (Generator)**: Qwen3-0.6B transformer with 32-step iterative unmasking and classifier-free guidance. Generates 8-codebook audio tokens from text.
- **Stage 1 (Decoder)**: HiggsAudioV2 RVQ quantizer + DAC acoustic decoder. Converts tokens to 24kHz waveform.

Both stages use `GPUGenerationWorker` with `OmniGenerationScheduler`.

## Notes

- Output audio is saved to `output.wav` by default. Use `--output` to change the path.
- The model estimates duration from text automatically via `RuleDurationEstimator`.
- Use `--stage-init-timeout` to increase the stage initialization timeout for first-time model downloads.
