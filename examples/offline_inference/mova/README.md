# MOVA Offline Inference

This example demonstrates offline video-audio generation using [MOVA](https://github.com/OpenMOSS/MOVA) (MOSS Video and Audio).

## Prerequisites

- `soundfile`: for audio WAV output (`pip install soundfile`)
- `ffmpeg`: for MP4 video+audio muxing (`apt install ffmpeg`)

## Usage

```bash
python examples/offline_inference/mova/end2end.py \
    --model /path/to/MOVA-360p \
    --prompt "a person talking and waving" \
    --ref-path reference.png \
    --enable-cpu-offload \
    --output mova_output.mp4
```

## Notes

- Currently supports MOVA-360p with I2VA (image-to-video-audio) mode.
- Default resolution: 352x640 (360p).
- `--enable-cpu-offload` is recommended for GPUs with less than ~80GB VRAM. Each video DiT is ~43GB (bf16), so two video DiTs + other components exceed single 48GB GPU memory. The pipeline swaps components between CPU and GPU during inference.
- With CPU offload, requires ~90GB system RAM (two 14B video DiTs + other components).
- Without CPU offload, all components must fit in GPU memory simultaneously (~80GB+).
