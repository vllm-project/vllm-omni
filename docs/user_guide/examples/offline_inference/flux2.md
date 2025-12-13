# FLUX 2 Offline Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/flux2>.

This folder provides two simple entrypoints for experimenting with `black-forest-labs/FLUX.2-dev` using vLLM-Omni:

- `text_to_image.py`: command-line script for single image generation.
- `gradio_demo.py`: lightweight Gradio UI for interactive prompt/seed/guidance exploration.

## Local CLI Usage

```bash
python text_to_image.py \
  --model black-forest-labs/FLUX.2-dev \
  --prompt "a serene mountain landscape at sunset" \
  --seed 42 \
  --guidance_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output outputs/flux2.png
```

Key arguments:

- `--prompt`: text description (string).
- `--seed`: integer seed for deterministic sampling.
- `--guidance_scale`: guidance strength (typically ~3.5–5.0).
- `--num_inference_steps`: diffusion sampling steps (more steps = higher quality, slower).
- `--height/--width`: output resolution.
- `--output`: path to save the generated PNG.

## Web UI Demo

Launch the gradio demo:

```bash
python gradio_demo.py --port 7863
```

## Example materials

??? abstract "gradio_demo.py"
    ``````py
    --8<-- "examples/offline_inference/flux2/gradio_demo.py"
    ``````
??? abstract "text_to_image.py"
    ``````py
    --8<-- "examples/offline_inference/flux2/text_to_image.py"
    ``````


