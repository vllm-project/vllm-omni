# AudioX offline inference

Generate audio with the [AudioX](https://zeyuet.github.io/AudioX/) MMDiT diffusion
pipeline (`AudioXPipeline`). Six tasks: `t2a`, `t2m`, `v2a`, `v2m`, `tv2a`, `tv2m`.

## Prerequisites

Download a vLLM-Omni weight bundle (component-sharded safetensors):

```bash
huggingface-cli download zhangj1an/AudioX --local-dir ./audiox_weights
```

The Hugging Face id `zhangj1an/AudioX` also works directly without prefetching.

## Usage

```bash
# Text-to-audio only (default uses zhangj1an/AudioX from the Hub):
python end2end.py --tasks t2a

# All six tasks against a local bundle and a sample video for v2*/tv2*:
python end2end.py \
  --model ./audiox_weights \
  --video https://zeyuet.github.io/AudioX/static/samples/V2M/1XeBotOFqHA.mp4

# Subset of tasks, custom seed and steps:
python end2end.py --tasks t2a tv2a --num-inference-steps 100 --seed 0
```

## Arguments

- `--model`: HF id or local bundle path (default: `zhangj1an/AudioX`).
- `--tasks`: any subset of `t2a t2m v2a v2m tv2a tv2m` (default: all).
- `--video`: video file/URL — required for `v2*` and `tv2*`.
- `--reference-audio`: optional audio prompt (audio-conditioned generation).
- `--num-inference-steps`, `--guidance-scale`, `--seed`, `--seconds-total`,
  `--sample-rate`, `--output-dir`: generation knobs.

Outputs land in `<output-dir>/<task>.wav` as 16-bit stereo WAV.
