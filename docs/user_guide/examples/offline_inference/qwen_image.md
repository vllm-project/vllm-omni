# Qwen-Image Offline Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen_image>.


This folder provides two simple entrypoints for experimenting with `Qwen/Qwen-Image` using vLLM-Omni:

- `text_to_image.py`: command-line script for single image generation.
- `web_demo.py`: lightweight Gradio UI for interactive prompt/seed/CFG exploration.


## Local CLI Usage

```bash
python text_to_image.py \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --cfg_scale 4.0 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --height 1024 \
  --width 1024 \
  --output outputs/coffee.png
```

Key arguments:

- `--prompt`: text description (string).
- `--seed`: integer seed for deterministic sampling.
- `--cfg_scale`: true CFG scale (model-specific guidance strength).
- `--num_images_per_prompt`: number of images to generate per prompt (saves as `output`, `output_1`, ...).
- `--num_inference_steps`: diffusion sampling steps (more steps = higher quality, slower).
- `--height/--width`: output resolution (defaults 1024x1024).
- `--output`: path to save the generated PNG.

> ℹ️ Qwen-Image currently publishes best-effort presets at `1328x1328`, `1664x928`, `928x1664`, `1472x1140`, `1140x1472`, `1584x1056`, and `1056x1584`. Adjust `--height/--width` accordingly for the most reliable outcomes.

## Web UI Demo

Launch the gradio demo:

```bash
python gradio_demo.py --port 7862
```

Then open `http://localhost:7862/` on your local browser to interact with the web UI.

## Example materials

??? abstract "gradio_demo.py"
    ``````py
    --8<-- "examples/offline_inference/qwen_image/gradio_demo.py"
    ``````
??? abstract "text_to_image.py"
    ``````py
    --8<-- "examples/offline_inference/qwen_image/text_to_image.py"
    ``````
