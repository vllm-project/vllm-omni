# Text-To-Audio

Generate audio from text prompts using vLLM-Omni's Stable Audio pipeline.

- `text_to_audio.py`: command-line script for WAV generation with Stable Audio Open.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Key Arguments](#key-arguments)
- [More CLI Examples](#more-cli-examples)
- [Advanced Features](#advanced-features)
- [FAQ](#faq)

## Overview

This folder provides an offline inference entrypoint for text-to-audio generation. The script accepts a text prompt, optional negative prompt, sampler settings, and parallelism controls, then saves the generated waveform as a WAV file.

### Supported Models

| Model | Audio Shape | Peak VRAM (GiB) | Model Weights (GiB) |
| ----- | ----------- | --------------- | ------------------- |
| `stabilityai/stable-audio-open-1.0` | 44.1 kHz WAV, up to about 47s | TBD | TBD |

!!! info

    Peak VRAM is based on basic single-card usage, batch size = 1, without acceleration or optimization features. Model weights are reported by the runtime log line `Model loading took xxx GiB and xxx seconds`.

Default model: `stabilityai/stable-audio-open-1.0`

## Prerequisites

`stabilityai/stable-audio-open-1.0` is a gated model. Before running the example:

1. Accept the model license on the [Hugging Face model page](https://huggingface.co/stabilityai/stable-audio-open-1.0).
2. Authenticate locally with Hugging Face:

```bash
huggingface-cli login
```

Install an audio writer if your environment does not already provide one:

```text
pip install soundfile
```

## Quick Start

### Python API

```python
import soundfile as sf

from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="stabilityai/stable-audio-open-1.0")
    outputs = omni.generate({
        "prompt": "The sound of a hammer hitting a wooden surface",
        "negative_prompt": "Low quality",
        "audio_start": 0.0,
        "audio_end": 10.0,
        "num_inference_steps": 100,
        "guidance_scale": 7.0,
        "num_waveforms": 1,
    })
    audio = outputs[0].request_output.audios[0]
    if audio.ndim == 3:
        audio = audio[0].T
    sf.write("stable_audio_output.wav", audio, 44100)
```

### Local CLI Usage

```bash
python text_to_audio.py \
  --model stabilityai/stable-audio-open-1.0 \
  --prompt "The sound of a hammer hitting a wooden surface" \
  --negative-prompt "Low quality" \
  --seed 42 \
  --guidance-scale 7.0 \
  --audio-length 10.0 \
  --num-inference-steps 100 \
  --output stable_audio_output.wav
```

## Key Arguments

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--model` | str | `"stabilityai/stable-audio-open-1.0"` | Model name or local model path |
| `--prompt` | str | `"The sound of a hammer hitting a wooden surface."` | Text description for audio generation |
| `--negative-prompt` | str | `"Low quality."` | Negative prompt for classifier-free guidance |
| `--seed` | int | `42` | Random seed for deterministic generation |
| `--guidance-scale` | float | `7.0` | Classifier-free guidance scale |
| `--audio-start` | float | `0.0` | Start time in seconds |
| `--audio-length` | float | `10.0` | Duration in seconds; Stable Audio Open supports up to about 47 seconds |
| `--num-inference-steps` | int | `100` | Diffusion sampling steps; more steps are usually higher quality and slower |
| `--num-waveforms` | int | `1` | Number of generated waveforms for the prompt |
| `--sample-rate` | int | `44100` | WAV sample rate |
| `--cache-backend` | str | `None` | Cache acceleration backend; Stable Audio currently supports `tea_cache` |
| `--tea-cache-rel-l1-thresh` | float | `0.2` | Relative L1 threshold for TeaCache |
| `--output` | str | `"stable_audio_output.wav"` | Path to save the generated WAV file |
| `--enable-cpu-offload` | flag | off | Enable model-wise CPU offloading to reduce GPU memory use |
| `--enable-layerwise-offload` | flag | off | Enable layer-wise CPU offloading to reduce GPU memory use |
| `--use-hsdp` | flag | off | Enable HSDP weight sharding for the Stable Audio DiT |
| `--hsdp-shard-size` | int | `1` | Number of GPUs used for HSDP sharding |
| `--hsdp-replicate-size` | int | `1` | Number of HSDP replica groups |
| `--tensor-parallel-size` | int | `1` | Number of GPUs used for tensor parallelism inside the DiT |
| `--ulysses-degree` | int | `1` | Number of GPUs used for Ulysses sequence parallelism |
| `--ulysses-mode` | str | `"strict"` | Ulysses mode: `strict` or `advanced_uaa` |
| `--ring-degree` | int | `1` | Number of GPUs used for ring sequence parallelism |
| `--cfg-parallel-size` | int | `1` | Set to `2` to enable classifier-free guidance parallelism |
| `--vae-patch-parallel-size` | int | `1` | Number of GPUs used for VAE patch or tile parallelism |

## More CLI Examples

### TeaCache Acceleration

```bash
python text_to_audio.py \
  --model stabilityai/stable-audio-open-1.0 \
  --prompt "A soft synth pad with gentle ocean waves" \
  --negative-prompt "Low quality" \
  --seed 42 \
  --guidance-scale 7.0 \
  --audio-length 10.0 \
  --num-inference-steps 100 \
  --cache-backend tea_cache \
  --output stable_audio_teacache.wav
```

### HSDP Multi-GPU Inference

To reduce per-GPU memory for multi-GPU inference, launch with HSDP:

```bash
python text_to_audio.py \
  --model stabilityai/stable-audio-open-1.0 \
  --prompt "The sound of a hammer hitting a wooden surface" \
  --negative-prompt "Low quality" \
  --seed 42 \
  --guidance-scale 7.0 \
  --audio-length 10.0 \
  --num-inference-steps 100 \
  --use-hsdp \
  --hsdp-shard-size 2 \
  --output stable_audio_hsdp.wav
```

## Advanced Features

### CPU Offloading

If the model does not fit in GPU memory, try model-wise or layer-wise CPU offloading:

```bash
python text_to_audio.py \
  --model stabilityai/stable-audio-open-1.0 \
  --prompt "Thunder rolling over a quiet forest" \
  --audio-length 10.0 \
  --enable-cpu-offload \
  --output stable_audio_cpu_offload.wav
```

### Parallelism

The script exposes tensor, Ulysses, ring, CFG, VAE patch, and HSDP parallelism controls. For shared diffusion parallelism concepts, see the vLLM-Omni parallelism documentation under `docs/user_guide/parallelism/`.

## FAQ

### Why do I get a Hugging Face access error?

Make sure you accepted the model license and authenticated with `huggingface-cli login` before running the script.

### How can I reduce CUDA out-of-memory errors?

Try shorter `--audio-length`, lower `--num-inference-steps`, `--enable-cpu-offload`, `--enable-layerwise-offload`, or HSDP on multiple GPUs.
