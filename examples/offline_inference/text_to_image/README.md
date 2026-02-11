# Text-To-Image

This folder provides several entrypoints for experimenting with `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512`, `Tongyi-MAI/Z-Image` (Base), and `Tongyi-MAI/Z-Image-Turbo` using vLLM-Omni:

- `text_to_image.py`: command-line script for single image generation with advanced options.
- `z_image_examples.py`: comparison examples showing Z-Image Base vs Turbo usage.
- `web_demo.py`: lightweight Gradio UI for interactive prompt/seed/CFG exploration.

Note that when you pass in multiple independent prompts, they will be processed sequentially. Batching requests is currently not supported.

## Basic Usage

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Qwen/Qwen-Image")
    prompt = "a cup of coffee on the table"
    outputs = omni.generate(prompt)
    images = outputs[0].request_output[0].images
    images[0].save("coffee.png")
```

Or put more than one prompt in a request.

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Qwen/Qwen-Image")
    prompts = [
      "a cup of coffee on a table",
      "a toy dinosaur on a sandy beach",
      "a fox waking up in bed and yawning",
    ]
    outputs = omni.generate(prompts)
    for i, output in enumerate(outputs):
      image = output.request_output[0].images[0].save(f"{i}.jpg")
```

!!! info

    However, it is not currently recommended to do so
    because not all models support batch inference,
    and batch requesting mostly does not provide significant performance improvement (despite the impression that it does).
    This feature is primarily for the sake of interface compatibility with vLLM and to allow for future improvements.

!!! info

    For diffusion pipelines, the stage config field `stage_args.[].runtime.max_batch_size` is 1 by default, and the input
    list is sliced into single-item requests before feeding into the diffusion pipeline. For models that do internally support
    batched inputs, you can [modify this configuration](../../../configuration/stage_configs.md) to let the model accept a longer batch of prompts.

Apart from string prompt, vLLM-Omni also supports dictionary prompts in the same style as vLLM.
This is useful for models that support negative prompts.

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Qwen/Qwen-Image")
    outputs = omni.generate([
      {
        "prompt": "a cup of coffee on a table"，
        "negative_prompt": "low resolution"
      },
      {
        "prompt": "a toy dinosaur on a sandy beach"，
        "negative_prompt": "cinematic, realistic"
      }
    ])
    for i, output in enumerate(outputs):
      image = output.request_output[0].images[0].save(f"{i}.jpg")
```

## Local CLI Usage

### Z-Image Turbo (Fast Inference)
```bash
python text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --num_images_per_prompt 1 \
  --num_inference_steps 8 \
  --guidance_scale 0.0 \
  --height 1024 \
  --width 1024 \
  --output outputs/coffee_turbo.png
```

### Z-Image Base (High Quality with CFG)
```bash
python text_to_image.py \
  --model Tongyi-MAI/Z-Image \
  --prompt "a cup of coffee on the table" \
  --negative_prompt "blurry, low quality, distorted" \
  --seed 42 \
  --num_images_per_prompt 1 \
  --num_inference_steps 50 \
  --guidance_scale 4.0 \
  --height 1280 \
  --width 720 \
  --output outputs/coffee_base.png
```

Key arguments:

- `--prompt`: text description (string).
- `--seed`: integer seed for deterministic sampling.
- `--cfg_scale`: true CFG scale (model-specific guidance strength).
- `--num_images_per_prompt`: number of images to generate per prompt (saves as `output`, `output_1`, ...).
- `--num_inference_steps`: diffusion sampling steps (more steps = higher quality, slower).
- `--height/--width`: output resolution (defaults 1024x1024).
- `--output`: path to save the generated PNG.
- `--vae_use_slicing`: enable VAE slicing for memory optimization.
- `--vae_use_tiling`: enable VAE tiling for memory optimization.
- `--cfg_parallel_size`: set it to 2 to enable CFG Parallel. See more examples in [`user_guide`](../../../docs/user_guide/diffusion/parallelism_acceleration.md#cfg-parallel).
- `--enable-cpu-offload`: enable CPU offloading for diffusion models.

> ℹ️ If you encounter OOM errors, try using `--vae_use_slicing` and `--vae_use_tiling` to reduce memory usage.

## Z-Image Base vs Turbo Comparison

For detailed comparison and usage examples of both Z-Image variants, see:

```bash
python z_image_examples.py --example all
```

Key differences:

| Feature | Z-Image Base | Z-Image Turbo |
|---------|--------------|---------------|
| Model | `Tongyi-MAI/Z-Image` | `Tongyi-MAI/Z-Image-Turbo` |
| Inference Steps | 28-50 (default: 50) | 8 |
| CFG Support | ✅ Yes (guidance_scale 3.0-5.0) | ❌ Must use 0.0 |
| Negative Prompts | ✅ Supported | ❌ Not supported |
| Fine-tunable | ✅ Yes | ❌ No (distilled) |
| Scheduler Shift | 6.0 | 3.0 |
| Best For | High quality, fine-tuning | Fast iteration, speed |

> ℹ️ Qwen-Image currently publishes best-effort presets at `1328x1328`, `1664x928`, `928x1664`, `1472x1140`, `1140x1472`, `1584x1056`, and `1056x1584`. Adjust `--height/--width` accordingly for the most reliable outcomes.

## Web UI Demo

Launch the gradio demo:

```bash
python gradio_demo.py --port 7862
```

Then open `http://localhost:7862/` on your local browser to interact with the web UI.
