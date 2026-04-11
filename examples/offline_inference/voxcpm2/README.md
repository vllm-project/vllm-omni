# VoxCPM2 Offline Inference (Native AR)

VoxCPM2 is a 2B-parameter tokenizer-free diffusion AR TTS model. It produces 48kHz audio and supports 30+ languages with a single-stage native AR pipeline backed by MiniCPM4.

## Prerequisites

Install the `voxcpm` package, or set the environment variable pointing to the source tree:

```bash
# Option A: install package
pip install voxcpm

# Option B: use source checkout
export VLLM_OMNI_VOXCPM_CODE_PATH=/path/to/voxcpm
```

## Quick Start

Zero-shot synthesis:

```bash
python examples/offline_inference/voxcpm2/end2end.py \
    --model openbmb/VoxCPM2 \
    --text "Hello, this is a VoxCPM2 demo." \
    --output-dir output_audio
```

Voice cloning with a reference audio:

```bash
python examples/offline_inference/voxcpm2/end2end.py \
    --text "Hello, this is a voice clone demo." \
    --reference-audio /path/to/reference.wav \
    --output-dir output_clone
```

Prompt continuation (matched audio + text prefix):

```bash
python examples/offline_inference/voxcpm2/end2end.py \
    --text "Continuation target sentence." \
    --prompt-audio /path/to/prompt.wav \
    --prompt-text "Transcript of the prompt audio." \
    --output-dir output_cont
```

The script accepts the following arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | `openbmb/VoxCPM2` | HuggingFace repo ID or local path |
| `--text` | (example sentence) | Text to synthesize |
| `--output-dir` | `output_audio` | Directory for output WAV files |
| `--stage-configs-path` | `voxcpm2.yaml` | Stage config YAML path |
| `--reference-audio` | `None` | Reference audio for voice cloning (isolated) |
| `--prompt-audio` | `None` | Prompt audio for continuation mode |
| `--prompt-text` | `None` | Transcript matching `--prompt-audio` |

## Performance

Measured on a single H20 GPU (80 GB), voxcpm 0.0.0, PyTorch 2.10.0+cu128:

| Input length | RTF | Sample rate |
|---|---|---|
| Short (~6 words) | ~0.81 | 48 kHz |
| Long (~50 words) | ~0.72 | 48 kHz |

RTF < 1.0 means faster than real time.

## Architecture

VoxCPM2 uses a single-stage native AR pipeline:

```
feat_encoder
└─► MiniCPM4 (base LM)
     └─► FSQ (finite scalar quantization)
          └─► residual_lm (residual AR)
               └─► LocDiT (local diffusion transformer)
                    └─► AudioVAE → 48 kHz waveform
```

All stages are fused into one vllm-native execution graph via `voxcpm2.yaml`, eliminating inter-stage coordination overhead and enabling true end-to-end batching.
