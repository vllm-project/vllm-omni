# Text-To-Image

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/text_to_image>.


This folder provides several entrypoints for experimenting with `Qwen/Qwen-Image` `Qwen/Qwen-Image-2512` `Tongyi-MAI/Z-Image-Turbo` using vLLM-Omni:

- `text_to_image.py`: command-line script for single image generation with advanced options.
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

```bash
python text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
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
    --8<-- "examples/offline_inference/text_to_image/gradio_demo.py"
    ``````
??? abstract "text_to_image.py"
    ``````py
    --8<-- "examples/offline_inference/text_to_image/text_to_image.py"
    ``````
