# HiDream-O1-Image (offline inference)

> **Status:** Phase 1 of the integration roadmap -- text-to-image only.
> Image editing, multi-reference personalization, and layout/skeleton
> conditioning are not yet supported; they land in later phases.

[HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-O1-Image) is a
Pixel-level Unified Transformer: a single Qwen3-VL-derived transformer that
performs flow-matching diffusion directly on raw pixel patches (no VAE, no
separate text encoder).

## Run

```bash
cd examples/offline_inference/hidream_o1_image
python text_to_image_hidream_o1.py \
    --model HiDream-ai/HiDream-O1-Image \
    --prompt "A cute cat sitting on a windowsill at sunset." \
    --height 1024 --width 1024 \
    --model-type full \
    --output output.png
```

For the distilled, faster-iterating variant:

```bash
python text_to_image_hidream_o1.py \
    --model HiDream-ai/HiDream-O1-Image-Dev \
    --prompt "A dog holds a sign that says \"hello\"." \
    --model-type dev \
    --output output_dev.png
```

## Key arguments

| Argument | Description |
| --- | --- |
| `--model-type` | `full` (50 steps, guidance 5.0, shift 3.0) or `dev` (28 steps, no CFG, shift 1.0). |
| `--num-inference-steps` / `--guidance-scale` / `--shift` | Override the `--model-type` preset. |
| `--scheduler` | `default` (`FlowUniPCMultistepScheduler`) or `flow_match` (diffusers `FlowMatchEulerDiscreteScheduler`). |
| `--height` / `--width` | Snapped to the nearest 32px-aligned resolution internally. |

## Requirements

- `transformers>=4.57.1` (for `Qwen3VLForConditionalGeneration`).
- A CUDA GPU. flash-attn is recommended but not required -- the pipeline
  builds a dense attention mask as a fallback for non-flash backends (set
  `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA` to force it).
