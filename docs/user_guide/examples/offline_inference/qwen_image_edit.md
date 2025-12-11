# Qwen-Image-Edit Offline Inference

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen_image>. Added in PR [#196](https://github.com/vllm-project/vllm-omni/pull/196).

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
- `--prompt` / `--negative_prompt`: describe what to add or avoid in the edit.
- `--cfg_scale`: true CFG scale for Qwen-Image-Edit (quality vs. fidelity).
- `--num_inference_steps`: diffusion sampling steps.
- `--num_outputs_per_prompt`: generate multiple edited variants in one run.

The script auto-detects CUDA/NPU, seeds a generator for reproducibility, and enables VAE tiling/slicing on NPUs.

## Example materials

??? abstract "image_edit.py"
    ``````py
    --8<-- "examples/offline_inference/qwen_image/image_edit.py"
    ``````

