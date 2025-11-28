# Qwen-Image Offline Inference

This folder provides two simple entrypoints for experimenting with `Qwen/Qwen-Image` using vLLM-Omni:

- `text_to_image.py`: command-line script for single image generation.
- `gradio_demo.py`: lightweight Gradio UI for interactive prompt/seed/CFG exploration.

## Prerequisites

Ensure you can run the Qwen-Image diffusion pipeline locally (CUDA GPU recommended). From the repo root:

```bash
pip install -e .[diffusion,gradio]
```

## CLI Usage (`text_to_image.py`)

```bash
python text_to_image.py \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output outputs/coffee.png
```

Key arguments:

- `--prompt`: text description (string).
- `--seed`: integer seed for deterministic sampling.
- `--cfg_scale`: true CFG scale (model-specific guidance strength).
- `--num_inference_steps`: diffusion sampling steps (more steps = higher quality, slower).
- `--height/--width`: output resolution (defaults 1024x1024).
- `--output`: path to save the generated PNG.

> ℹ️ Qwen-Image currently publishes best-effort presets at `1328x1328`, `1664x928`, `928x1664`, `1472x1140`, `1140x1472`, `1584x1056`, and `1056x1584`. Adjust `--height/--width` accordingly for the most reliable outcomes.

## Gradio Demo (`gradio_demo.py`)

Launch an interactive UI:

```bash
python gradio_demo.py --ip 0.0.0.0 --port 7862
```

Supported presets (width × height):

- `1328 x 1328` (1:1)
- `1664 x 928` (16:9)
- `928 x 1664` (9:16)
- `1472 x 1140` (4:3)
- `1140 x 1472` (3:4)
- `1584 x 1056` (3:2)
- `1056 x 1584` (2:3)

Pick the resolution directly from the dropdown (CLI defaults must match one of the presets).

UI features:

- Prompt textbox with customizable default (`--default-prompt`).
- Numeric inputs for seed, CFG scale, and inference steps (`--default-seed`, `--default-cfg-scale`, `--num_inference_steps`).
- Download button below the rendered image.

Use `--share` for a public Gradio link when needed. Adjust `--model`, `--height`, and `--width` to point to custom checkpoints or resolutions.

