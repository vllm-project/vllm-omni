# JoyAI-Echo (offline inference, single-shot)

[JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo) is JD's
minute-level multi-shot audio-video generation model built on top of
LTX-2.3. This example covers **single-shot** text → video+audio
inference using the generic
[`examples/offline_inference/text_to_video/text_to_video.py`](../text_to_video/text_to_video.py)
launcher (the script auto-detects a `joyai` preset from the model id /
local path). Multi-shot generation with the paired audio-video memory
bank is planned for a follow-up PR.

See [`recipes/JD/JoyAI-Echo.md`](../../../recipes/JD/JoyAI-Echo.md) for
the full recipe (parameters, validated configs, and limitations).

## Prerequisites

- One H100 / A100 / B200 (≥ 80 GB) or equivalent GPU. Peak VRAM is
  ~68 GB on B200 with the default 121-frame 480 × 832 setting (PR1
  keeps text encoder / VAEs / vocoder GPU-resident; PR2 will reintroduce
  CPU offload to recover the ~28 GB gap vs upstream).
- `ffmpeg >= 6` available on `PATH` (used to mux mp4+audio).
- vllm-omni built from source.

## Download the checkpoints

JoyAI-Echo requires **two separate downloads**: the JoyAI-Echo
checkpoint (which ships only the monolithic safetensors + minimal
metadata) and a standalone Gemma-3-12B-IT checkpoint that the upstream
inference reference loads via its own ``paths.gemma_path`` config field.

> **Important — the model must be pre-downloaded.** ``JoyAIEchoPipeline``
> reads the monolithic safetensors file directly via ``safe_open``; passing
> a bare Hub ID (e.g. ``jdopensource/JoyAI-Echo``) for ``--model`` will fail
> with ``FileNotFoundError``. Always download to a local directory first
> using ``huggingface-cli download`` (or any equivalent mirror) and pass
> the absolute local path to ``--model``.

```bash
# 1) JoyAI-Echo monolithic safetensors + metadata stubs (~46 GB)
huggingface-cli download jdopensource/JoyAI-Echo --local-dir ./JoyAI-Echo

# 2) Gemma-3-12B-IT (separate download, ~24 GB)
#    Gated repo — accept the licence on the model page and run
#    `huggingface-cli login` first.
huggingface-cli download google/gemma-3-12b-it --local-dir ./gemma-3-12b-it
```

The JoyAI-Echo directory is expected to contain (current upstream
layout, post review):

```text
JoyAI-Echo/
├── JoyAI-Echo-release.safetensors
├── model_index.json                # shipped by upstream
├── config.json                     # shipped by upstream
└── transformer/
    └── config.json                 # shipped by upstream
```

The Gemma-3-12B-IT directory is the standard HF release:

```text
gemma-3-12b-it/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── model-0000{1..5}-of-00005.safetensors
```

### Pointing JoyAI-Echo at Gemma-3-12B-IT

The pipeline resolves the Gemma directory in this order:

1. **Environment variable** ``JOYAI_ECHO_GEMMA_PATH`` (recommended) — set
   to the absolute path of the Gemma-3-12B-IT directory.
2. **Subfolder fallback** — if the JoyAI-Echo directory contains a
   ``text_encoder/`` sub-directory with valid Gemma-3 files, that is
   used (kept for backward compatibility with mirrors that bundle the
   two checkpoints together).
3. Otherwise the pipeline raises ``FileNotFoundError`` with the two
   above options listed.

The recommended setup is option (1):

```bash
export JOYAI_ECHO_GEMMA_PATH=/abs/path/to/gemma-3-12b-it
```

## Run

```bash
export JOYAI_ECHO_GEMMA_PATH=/abs/path/to/gemma-3-12b-it

python examples/offline_inference/text_to_video/text_to_video.py \
    --model /abs/path/to/JoyAI-Echo \
    --model-class-name JoyAIEchoPipeline \
    --prompt "A cute orange cat sitting on a sofa, soft natural light, gentle ambient music." \
    --output joyai_echo_output.mp4
```

The `joyai` preset (auto-selected when the model path contains `joyai`)
provides the DMD-distilled defaults (`height=480`, `width=832`,
`num_frames=121`, `num_inference_steps=8`, `guidance_scale=1.0`,
`fps=25`, `frame_rate=25`). Override any of these with the standard CLI
flags on `text_to_video.py`.

## Default parameters

The DMD-distilled schedule provides 8-step inference (9 sigmas) for a
~7.5x speedup over LTX-2.3's 30-step baseline:

| Parameter | Default | Notes |
|---|---|---|
| `denoising_sigmas` | `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]` | Override via `OmniDiffusionSamplingParams.sigmas` |
| `height x width` | 480 × 832 | Must be divisible by 32 |
| `num_frames` | 121 | Must satisfy `(num_frames - 1) % 8 == 0` |
| `fps` / `frame_rate` | 25 | LTX-2.3 native rate |
| `num_inference_steps` | 8 | DMD few-step schedule |
| `guidance_scale` | 1.0 | DMD-distilled, CFG disabled |
| `dtype` | `bfloat16` | hardcoded in pipeline |

## Limitations (PR1 scope)

- **Single-shot only.** Maximum duration is bounded by `num_frames`
  (e.g. 121 frames @ 25fps ≈ 4.84s). Multi-shot, minute-/5-minute-level
  generation requires the `PairedAudioVideoMemoryBank` and is the scope
  of PR2.
- **No LoRA support yet** (the upstream config supports a memory LoRA;
  port deferred).
- **No tensor / sequence / CFG parallelism** (PR3).
- **License**: JoyAI-Echo is under the LTX-2 Community License
  (academic / non-commercial). Read the `LICENSE` and
  `THIRD_PARTY_NOTICES.md` files inside the model repo before using.
