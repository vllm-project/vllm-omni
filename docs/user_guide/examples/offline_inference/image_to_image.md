# Image-To-Image

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/image_to_image>.


This example edits an input image with `Qwen/Qwen-Image-Edit` using the `image_edit.py` CLI.

## Local CLI Usage

```bash
python image_edit.py \
  --image qwen_bear.png \
  --prompt "Let this mascot dance under the moon, surrounded by floating stars and poetic bubbles such as 'Be Kind'" \
  --output output_image_edit.png \
  --num_inference_steps 50 \
  --cfg_scale 4.0
```

Key arguments:

- `--image`: path to the source image (PNG/JPG, converted to RGB).
- `--prompt` / `--negative_prompt`: text description (string).
- `--cfg_scale`: true CFG scale for Qwen-Image-Edit (quality vs. fidelity).
- `--num_inference_steps`: diffusion sampling steps (more steps = higher quality, slower).
- `--output`: path to save the generated PNG.

## Example materials

??? abstract "image_edit.py"
    ``````py
    --8<-- "examples/offline_inference/image_to_image/image_edit.py"
    ``````
