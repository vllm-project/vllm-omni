# Image-To-Image

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_image>.


This example edits an input image with `Qwen/Qwen-Image-Edit` using the `image_edit.py` CLI.

## Local CLI Usage

### Single Image Editing

Download the example image:

```bash
wget https://vllm-public-assets.s3.us-west-2.amazonaws.com/omni-assets/qwen-bear.png
```

Then run:

```bash
python image_edit.py \
  --image qwen-bear.png \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output output_image_edit.png \
  --num_inference_steps 50 \
  --cfg_scale 4.0
```

### Multiple Image Editing (Qwen-Image-Edit-2509)

For multiple image inputs, use `Qwen/Qwen-Image-Edit-2509` or  `Qwen/Qwen-Image-Edit-2511`:

```bash
python image_edit.py \
  --model Qwen/Qwen-Image-Edit-2509 \
  --image img1.png img2.png \
  --prompt "Combine these images into a single scene" \
  --output output_image_edit.png \
  --num_inference_steps 50 \
  --cfg_scale 4.0 \
  --guidance_scale 1.0
```

Key arguments:

- `--model`: model name or path. Use `Qwen/Qwen-Image-Edit-2509` or later for multiple image support.
- `--image`: path(s) to the source image(s) (PNG/JPG, converted to RGB). Can specify multiple images.
- `--prompt` / `--negative_prompt`: text description (string).
- `--cfg_scale`: true classifier-free guidance scale (default: 4.0). Classifier-free guidance is enabled by setting cfg_scale > 1 and providing a negative_prompt. Higher guidance scale encourages images closely linked to the text prompt, usually at the expense of lower image quality.
- `--guidance_scale`: guidance scale for guidance-distilled models (default: 1.0, disabled). Unlike classifier-free guidance (--cfg_scale), guidance-distilled models take the guidance scale directly as an input parameter. Enabled when guidance_scale > 1. Ignored when not using guidance-distilled models.
- `--num_inference_steps`: diffusion sampling steps (more steps = higher quality, slower).
- `--output`: path to save the generated PNG.

## Example materials

??? abstract "image_edit.py"
    ``````py
    --8<-- "examples/offline_inference/image_to_image/image_edit.py"
    ``````
??? abstract "run_qwen_image_edit_2511.sh"
    ``````sh
    --8<-- "examples/offline_inference/image_to_image/run_qwen_image_edit_2511.sh"
    ``````
