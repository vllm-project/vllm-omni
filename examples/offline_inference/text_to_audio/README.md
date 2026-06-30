# Text-To-Audio

vLLM-Omni supports diffusion-based audio generation models. Each model
lives in its own subdirectory and ships a single `end2end.py` script;
this README is the single doc entry point for offline inference of all
of them.

For online serving, see [`examples/online_serving/text_to_audio/`](../../online_serving/text_to_audio/README.md).
For the full list of supported architectures across all modalities, see
[Supported Models](../../../docs/models/supported_models.md).

For text-to-speech (autoregressive TTS), see [`examples/offline_inference/text_to_speech/`](../text_to_speech/README.md).

## Supported Models

| Model | HuggingFace repo | Tasks | Pipeline | Sample rate |
|---|---|---|---|---|
| Stable Audio Open | `stabilityai/stable-audio-open-1.0` | `t2a` (text → audio) | DiT diffusion | 44.1 kHz |
| AudioX | `zhangj1an/AudioX` | `t2a`, `t2m`, `v2a`, `v2m`, `tv2a`, `tv2m` | MMDiT diffusion | 48 kHz (default) |

## Common Quick Start

```bash
python examples/offline_inference/text_to_audio/<model>/end2end.py \
    --prompt "The sound of a hammer hitting a wooden surface"
```

Per-model flags (HSDP / TP / cache backend / video conditioning) are
documented in each model's section below.

---

## Stable Audio Open

`stabilityai/stable-audio-open-1.0` is a DiT diffusion pipeline that
generates audio from text prompts.

### Prerequisites

The model is gated on Hugging Face:

1. Accept the user agreement on the model page.
2. Authenticate locally:
   ```bash
   huggingface-cli login
   ```

### Quick start

```bash
python examples/offline_inference/text_to_audio/stable_audio/end2end.py \
    --model stabilityai/stable-audio-open-1.0 \
    --prompt "The sound of a hammer hitting a wooden surface" \
    --negative-prompt "Low quality" \
    --seed 42 \
    --guidance-scale 7.0 \
    --audio-length 10.0 \
    --num-inference-steps 100 \
    --cache-backend tea_cache \
    --output stable_audio_output.wav
```

### Multi-GPU (HSDP)

To reduce per-GPU memory, launch with HSDP weight sharding:

```bash
python examples/offline_inference/text_to_audio/stable_audio/end2end.py \
    --model stabilityai/stable-audio-open-1.0 \
    --prompt "The sound of a hammer hitting a wooden surface" \
    --negative-prompt "Low quality" \
    --seed 42 \
    --guidance-scale 7.0 \
    --audio-length 10.0 \
    --num-inference-steps 100 \
    --use-hsdp \
    --hsdp-shard-size 2 \
    --output stable_audio_output.wav
```

### Key arguments

- `--prompt`: text description.
- `--negative-prompt`: negative prompt for classifier-free guidance.
- `--seed`: integer seed for deterministic generation.
- `--guidance-scale`: classifier-free guidance scale.
- `--audio-length`: audio duration in seconds (max ~47 s for `stable-audio-open-1.0`).
- `--num-inference-steps`: diffusion sampling steps (more steps = higher quality, slower).
- `--use-hsdp`: enable HSDP weight sharding for the DiT.
- `--hsdp-shard-size`: number of GPUs used for HSDP sharding.
- `--hsdp-replicate-size`: number of HSDP replica groups.
- `--cache-backend`: cache acceleration backend. Stable Audio currently supports `tea_cache`.
- `--output`: path to save the generated WAV file.

---

## AudioX

[AudioX](https://zeyuet.github.io/AudioX/) is an MMDiT diffusion pipeline
(`AudioXPipeline`) covering six tasks: `t2a`, `t2m`, `v2a`, `v2m`,
`tv2a`, `tv2m`.

### Prerequisites

Download a vLLM-Omni weight bundle (component-sharded safetensors):

```bash
huggingface-cli download zhangj1an/AudioX --local-dir ./audiox_weights
```

The Hugging Face id `zhangj1an/AudioX` also works directly without prefetching.

### Quick start

```bash
# Text-to-audio only (default uses zhangj1an/AudioX from the Hub):
python examples/offline_inference/text_to_audio/audiox/end2end.py --tasks t2a

# All six tasks against a local bundle, with a sample video for v2*/tv2*:
python examples/offline_inference/text_to_audio/audiox/end2end.py \
    --model ./audiox_weights \
    --video https://zeyuet.github.io/AudioX/static/samples/V2M/1XeBotOFqHA.mp4

# Subset of tasks, custom seed and steps:
python examples/offline_inference/text_to_audio/audiox/end2end.py \
    --tasks t2a tv2a --num-inference-steps 100 --seed 0
```

### Key arguments

- `--model`: HF id or local bundle path (default: `zhangj1an/AudioX`).
- `--tasks`: any subset of `t2a t2m v2a v2m tv2a tv2m` (default: all).
- `--video`: video file / URL — required for `v2*` and `tv2*`.
- `--reference-audio`: optional audio prompt (audio-conditioned generation).
- `--num-inference-steps`, `--guidance-scale`, `--seed`, `--seconds-total`,
  `--sample-rate`, `--output-dir`: generation knobs.

Outputs land in `<output-dir>/<task>.wav` as 16-bit stereo WAV.
