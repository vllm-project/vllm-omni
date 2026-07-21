# JoyAI-Echo

> JD's minute-level multi-shot audio-video generation model, built on top
> of LTX-2.3 with a paired audio-video memory bank. PR1 (this recipe)
> covers single-shot text → video+audio offline inference.

## Summary

- Vendor: JD (jd-opensource)
- Model: [`jdopensource/JoyAI-Echo`](https://github.com/jd-opensource/JoyAI-Echo)
- Task: Text → video + synchronized audio
- Mode: Offline inference (online serving lands in a follow-up PR)
- Maintainer: @jefferrrrrrrrrrry

## When to use this recipe

Use this recipe to generate a single shot of synchronized
video + audio from a text prompt. The pipeline subclasses
`LTX23Pipeline` and replaces the diffusion transformer with a
JoyAI-Echo–specific stack (DMD-distilled 8-step schedule, dual
audio/video VAEs, BWE vocoder).

PR1 is **single-shot only**: the maximum duration is bounded by the
`num_frames` you request (e.g. 121 frames @ 25fps ≈ 4.84s). Multi-shot
minute-/5-minute-level generation requires the
`PairedAudioVideoMemoryBank` and is the scope of PR2; the official
upstream test prompts (`prompts/test_*.json` in the JoyAI-Echo repo)
each describe 12–15 sequential shots that PR1 will only render the
first of.

## References

- Upstream model & weights: <https://github.com/jd-opensource/JoyAI-Echo>
- HF mirror: `jdopensource/JoyAI-Echo`
- vLLM-Omni RFC: <https://github.com/vllm-project/vllm-omni/issues/4193>

## Prerequisites

- One H100 / A100 / B200 (≥ 80 GB) or equivalent. Peak VRAM is ~76 GB on
  B200 with the default 480 × 832 × 121 setting (PR1 keeps the text
  encoder + VAEs + vocoder GPU-resident; PR2 will reintroduce CPU
  offload).
- `ffmpeg >= 6` on `PATH` (used to mux mp4 + audio).
- `vllm-omni` built from source.

JoyAI-Echo requires **two separate downloads**: the JoyAI-Echo
checkpoint (which only ships the monolithic safetensors + minimal
metadata stubs) and a standalone Gemma-3-12B-IT checkpoint that the
upstream inference reference loads via its own ``paths.gemma_path``
config field. The pipeline cannot be loaded from a bare Hub ID:
``JoyAIEchoPipeline`` reads the safetensors via ``safe_open`` and
expects an absolute local path.

```bash
# 1) JoyAI-Echo monolithic safetensors + metadata stubs (~46 GB)
huggingface-cli download jdopensource/JoyAI-Echo --local-dir ./JoyAI-Echo

# 2) Gemma-3-12B-IT (separate gated download, ~24 GB)
#    Accept the licence on the model page and run `huggingface-cli login`
#    first.
huggingface-cli download google/gemma-3-12b-it --local-dir ./gemma-3-12b-it
```

The JoyAI-Echo directory is expected to contain (current upstream
layout, post-review):

```text
JoyAI-Echo/
├── JoyAI-Echo-release.safetensors
├── model_index.json                # shipped by upstream
├── config.json                     # shipped by upstream
└── transformer/
    └── config.json                 # shipped by upstream
```

The Gemma-3-12B-IT directory is the standard HF release (``config.json``
+ ``tokenizer*`` + 5 sharded ``model-*.safetensors`` files).

### Pointing JoyAI-Echo at Gemma-3-12B-IT

The pipeline resolves the Gemma checkpoint in this order:

1. ``$JOYAI_ECHO_GEMMA_PATH`` — recommended. Mirrors the upstream
   ``paths.gemma_path`` configuration. Set to the absolute path of the
   Gemma-3-12B-IT directory.
2. ``<model>/text_encoder/`` — backward-compatible subfolder fallback
   for mirrors that bundle the two checkpoints together.
3. Otherwise the pipeline raises ``FileNotFoundError`` listing both
   options.

## Offline inference

### Command

The generic `text_to_video.py` launcher auto-detects a `joyai` preset
from the model id / local path. Pass `--model-class-name
JoyAIEchoPipeline` so Omni dispatches to the JoyAI-Echo pipeline.

```bash
export JOYAI_ECHO_GEMMA_PATH=/abs/path/to/gemma-3-12b-it

python examples/offline_inference/text_to_video/text_to_video.py \
    --model /abs/path/to/JoyAI-Echo \
    --model-class-name JoyAIEchoPipeline \
    --prompt "A cute orange cat sitting on a sofa, soft natural light, gentle ambient music." \
    --output joyai_echo_output.mp4
```

### Default parameters (joyai preset)

| Parameter | Default | Notes |
|---|---|---|
| `height × width` | 480 × 832 | Both must be divisible by 32 |
| `num_frames` | 121 | Must satisfy `(num_frames - 1) % 8 == 0` |
| `num_inference_steps` | 8 | DMD few-step schedule (9 sigmas) |
| `guidance_scale` | 1.0 | DMD distilled — CFG disabled |
| `fps` | 25 | LTX-2.3 native rate |
| `frame_rate` | 25 | Forwarded to `OmniDiffusionSamplingParams` |
| `dtype` | `bfloat16` | Hardcoded in the pipeline |
| `denoising_sigmas` | `[1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]` | Override via `OmniDiffusionSamplingParams.sigmas` |

Override any of these with the standard CLI flags on `text_to_video.py`
(`--height`, `--num-frames`, `--num-inference-steps`, …).

### Notes

- Output is an mp4 with synchronized audio (24 kHz mono via the BWE
  vocoder).
- Memory: weights load at ~70 GB (JoyAI 46 GB + Gemma 24 GB); peak
  inference VRAM is ~76 GB on a single B200 with the default
  480 × 832 × 121 setting. The upstream reference inference script
  measures ~40 GB peak on the same setting by aggressively offloading
  components between calls; PR1 keeps ``text_encoder`` / ``vae`` /
  ``audio_vae`` / ``vocoder`` resident on GPU after the parent's first
  ``.to(device)`` swap. PR2 will integrate the existing
  ``--enable-cpu-offload`` plumbing to recover that budget.
- The joyai preset uses `frame_rate = 25`, which is required by the
  LTX-2.3-style transformer. The launcher forwards this through
  `OmniDiffusionSamplingParams.frame_rate`.

### Limitations (PR1 scope)

- **Single-shot only** — multi-shot pipelines, the
  `PairedAudioVideoMemoryBank`, and the LoRA hooks land in PR2.
- **No tensor / sequence / CFG parallelism** — PR3.
- **No online serving** — a follow-up PR will register the
  `JoyAIEchoPipeline` with `vllm serve --omni`.
- **License**: JoyAI-Echo is released under the LTX-2 Community License
  (academic / non-commercial). Read the upstream `LICENSE` and
  `THIRD_PARTY_NOTICES.md` before integrating downstream.

## Hardware Support

### GPU

#### 1× NVIDIA B200 (180 GB)

##### Environment

- OS: Ubuntu 22.04
- Python: 3.12
- Driver / runtime: CUDA 13.0
- vLLM: 0.22.0 (tag [`v0.22.0`](https://github.com/vllm-project/vllm/releases/tag/v0.22.0))
- vLLM-Omni: built from source on this PR branch (`joyai-echo-pr1`, rebased on `main`)

##### Validated configurations

| Duration | Frames | Resolution | Steps | Guidance | Inference Time | Peak VRAM |
|----------|--------|------------|-------|----------|----------------|-----------|
| 4.84s | 121 | 480×832 | 8 | 1.0 | ~25 s | ~76 GiB |
