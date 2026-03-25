# Fish Speech S2 Pro

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/fish_speech>.


This directory contains an offline demo for running Fish Speech S2 Pro (`fishaudio/s2-pro`) with vLLM Omni. It supports text-only synthesis and voice cloning with reference audio.

## Model Overview

[Fish Speech S2 Pro](https://huggingface.co/fishaudio/s2-pro) is a 4B dual-AR text-to-speech model by FishAudio. It uses:

- **Slow AR**: Generates semantic tokens autoregressively (Qwen3-based backbone)
- **Fast AR**: Predicts residual codebook tokens from semantic tokens
- **DAC Decoder**: Converts codec codes to 44.1 kHz audio waveform

The model produces high-quality speech with voice cloning capabilities.

## Prerequisites

Install the `fish-speech` package for the DAC codec:

```bash
pip install fish-speech
```

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

## Quick Start

### Text-Only Synthesis

```bash
python end2end.py --text "Hello, this is a test of the Fish Speech text to speech system."
```

Generated audio files are saved to `output_audio/` by default.

### Voice Cloning

Provide a reference audio file and its transcript to clone a voice:

```bash
python end2end.py \
    --text "Hello, this is a cloned voice." \
    --ref-audio /path/to/reference.wav \
    --ref-text "Transcript of the reference audio."
```

## Streaming Mode

Add `--streaming` to stream audio chunks progressively via `AsyncOmni` (requires `async_chunk: true` in the stage config):

```bash
python end2end.py --text "Hello, this is a test." --streaming
```

Each DAC decoder chunk is logged as it arrives. The final WAV file is written once generation completes.

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--text` | `"Hello, this is a test..."` | Text to synthesize |
| `--ref-audio` | None | Path to reference audio for voice cloning |
| `--ref-text` | None | Transcript of the reference audio |
| `--model` | `fishaudio/s2-pro` | HuggingFace model path |
| `--stage-configs-path` | `fish_speech_s2_pro.yaml` | Path to stage configs YAML |
| `--output-dir` | `output_audio` | Output directory for WAV files |
| `--streaming` | False | Enable streaming via AsyncOmni |
| `--stage-init-timeout` | 600 | Stage initialization timeout (seconds) |

## Notes

- Output audio is 44.1 kHz mono WAV.
- The DAC codec weights (`codec.pth`) are loaded lazily from the model directory.
- Voice cloning encodes the reference audio using the DAC codec to extract semantic codes, then prepends them as a system message.

## Example materials

??? abstract "end2end.py"
    ``````py
    --8<-- "examples/offline_inference/fish_speech/end2end.py"
    ``````
