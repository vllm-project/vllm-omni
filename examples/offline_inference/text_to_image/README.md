# Text-To-Image

Generate images from text prompts using vLLM-Omni's diffusion pipeline entrypoints.

- `text_to_image.py`: command-line script for single image generation with advanced options.
- `gradio_demo.py`: lightweight Gradio UI for interactive prompt/seed/CFG exploration.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Key Arguments](#key-arguments)
- [More CLI Examples](#more-cli-examples)
- [Web UI Demo](#web-ui-demo)

## Overview

This folder provides several entrypoints for experimenting with text-to-image diffusion models using vLLM-Omni. Note that `NextStep-1.1` has a different architecture, so it is treated differently regarding running arguments and pipeline.

### Supported Models

| Model | Image Shape  | Peak VRAM (GiB) * | Model Weights (GiB) |
| ----- | ----------- | ----------- | ----------------- |
| `Qwen/Qwen-Image` | 1024 x 1024 | 60.0 | 53.7 |
| `Qwen/Qwen-Image-2512` |1024 x 1024 | 60.0 | 53.7 |
| `Tongyi-MAI/Z-Image-Turbo` | 1024 x 1024 | 24.8 | 19.2 |
| `stepfun-ai/NextStep-1.1` | 512 x 512 | 71.8 | 28.1 |
| `meituan-longcat/LongCat-Image` | 1024 x 1024 | 71.2 | 27.3 |
| `AIDC-AI/Ovis-Image-7B` | 1024 x 1024 | 71.8 | 17.1 |
| `OmniGen2/OmniGen2` |  1024 x 1024 | 20.1 | 14.7 |
| `stabilityai/stable-diffusion-3.5-medium` | 1024 x 1024 | 20.1 | 15.6 |
| `black-forest-labs/FLUX.1-dev` | 1024 x 1024 | 33.9 | 31.4 |
| `black-forest-labs/FLUX.1-schnell` | 1024 x 1024 | 33.9 | 31.4 |
| `black-forest-labs/FLUX.2-klein-4B` | 1024 x 1024 | 72.7 | 14.9 |
| `black-forest-labs/FLUX.2-klein-9B` | 1024 x 1024 | 37.1 | 32.3 |
| `black-forest-labs/FLUX.2-dev` | 1024 x 1024 | 65.7 | >80 (CPU offload required) |
| `HunyuanImage-3.0` | 1024 x 1024 | 80.0 (TP≥3)  | 160 |

!!! info
*Peak VRAM:  based on basic single-card usage, batch size =1, without any acceleration/optimization features. FLUX.2-dev requires `--enable-cpu-offload` on a single 80 GiB GPU.

Default model: `Qwen/Qwen-Image`

## Quick Start

### Python API

Single-prompt generation:

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Qwen/Qwen-Image")
    prompt = "a cup of coffee on the table"
    outputs = omni.generate(prompt)
    images = outputs[0].request_output.images
    images[0].save("coffee.png")
```

### Local CLI Usage

```bash
python text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "a cup of coffee on the table" \
  --output coffee.png
```

## Key Arguments

**Common arguments:**

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--prompt` | str | `"a cup of coffee on the table"` | Text description for image generation |
| `--seed` | int | `142` | Integer seed for deterministic sampling |
| `--negative-prompt` | str | `None` | Negative prompt for classifier-free conditional guidance |
| `--cfg-scale` | float | `4.0` | True CFG scale (model-specific guidance strength) |
| `--guidance-scale` | float | `1.0` | Classifier-free guidance scale |
| `--num-images-per-prompt` | int | `1` | Number of images per prompt (saved as `output`, `output_1`, ...) |
| `--num-inference-steps` | int | `50` | Diffusion sampling steps (more steps = higher quality, slower) |
| `--height` | int | `1024` | Output image height in pixels |
| `--width` | int | `1024` | Output image width in pixels |
| `--output` | str | `"qwen_image_output.png"` | Path to save the generated image |
| `--vae-use-slicing` | flag | off | Enable VAE slicing for memory optimization |
| `--vae-use-tiling` | flag | off | Enable VAE tiling for memory optimization |
| `--cfg-parallel-size` | int | `1` | Set to `2` to enable CFG Parallel |
| `--ulysses-degree` | int | `1` | Ulysses sequence parallel degree for multi-GPU inference |
| `--ring-degree` | int | `1` | Ring sequence parallel degree for hybrid Ulysses + Ring inference |
| `--ulysses-mode` | str | `"strict"` | Ulysses SP mode: `"strict"` or `"advanced_uaa"` |
| `--enable-cpu-offload` | flag | off | Enable CPU offloading for diffusion models |
| `--lora-path` | str | — | Path to PEFT LoRA adapter folder |
| `--lora-scale` | float | `1.0` | Scale factor for LoRA weights |
| `--use-system-prompt` | str | `None` | System prompt preset: `en_unified`, `en_vanilla`, `en_recaption`, `en_think_recaption`, `dynamic`, `None`, or custom text. Recommended: `en_unified`. Only for HunyuanImage-3.0.|
| `--system-prompt` | str | `None` | Custom system prompt text. Only used when `--use-system-prompt` is set to `custom`. Only for HunyuanImage-3.0.|

**NextStep-1.1 specific arguments:**

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--guidance-scale-2` | float | `1.0` | Secondary guidance scale (e.g. image-level CFG) |
| `--timesteps-shift` | float | `1.0` | Timesteps shift parameter for sampling |
| `--cfg-schedule` | str | `"constant"` | CFG schedule type: `"constant"` or `"linear"` |
| `--use-norm` | flag | off | Apply layer normalization to sampled tokens |

> If you encounter OOM errors, try using `--vae-use-slicing` and `--vae-use-tiling` to reduce memory usage.

> Qwen-Image currently publishes best-effort presets at `1328x1328`, `1664x928`, `928x1664`, `1472x1140`, `1140x1472`, `1584x1056`, and `1056x1584`. Adjust `--height/--width` accordingly for the most reliable outcomes.

## More CLI Examples

### Tongyi Models

```bash
python text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "a cup of coffee on the table" \
  --seed 42 \
  --guidance-scale 0.0 \
  --num-images-per-prompt 1 \
  --num-inference-steps 9 \
  --height 1024 \
  --width 1024 \
  --output outputs/coffee.png
```

`Tongyi-MAI/Z-Image-Turbo` is a distilled version of Z-Image. Distilled diffusion models usually require less number of inference steps (4~9), and Classifier-Free Guidance (CFG) is usually NOT applied. Similar distilled models are `black-forest-labs/FLUX.2-klein-4B` and `black-forest-labs/FLUX.2-klein-9B`.

Advanced UAA example (requires 2 GPUs):

```bash
python text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "a cup of coffee on the table" \
  --ulysses-degree 2 \
  --ulysses-mode advanced_uaa \
  --height 1024 \
  --width 1024 \
  --output outputs/coffee_hybrid.png
```

### NextStep Models

NextStep-1.1 supports extra arguments for dual-level CFG control:

```bash
python text_to_image.py \
  --model stepfun-ai/NextStep-1.1 \
  --prompt "A baby panda wearing an Iron Man mask, holding a board with 'NextStep-1' written on it" \
  --height 512 \
  --width 512 \
  --num-inference-steps 28 \
  --guidance-scale 7.5 \
  --guidance-scale-2 1.0 \
  --cfg-schedule constant \
  --output nextstep_output.png \
  --seed 42
```

### FLUX.2-dev Models

To run FLUX.2-dev on a single GPU, `--enable-cpu-offload` is required because the model weights exceed 80 GiB:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model black-forest-labs/FLUX.2-dev \
  --prompt "a lovely bunny holding a sign that says 'vllm-omni'" \
  --seed 42 \
  --tensor-parallel-size 1 \
  --num-images-per-prompt 1 \
  --num-inference-steps 50 \
  --guidance-scale 4.0 \
  --height 1024 \
  --width 1024 \
  --enable-cpu-offload \
  --output flux2-dev.png
```

### Batch Requests (Multiple Prompts)

You can pass multiple prompts in a single `generate` call.

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
        output.request_output.images[0].save(f"{i}.jpg")
```

!!! info

    Not all models support batch inference, and batch requesting mostly does not provide significant
    performance improvement. This feature is primarily for interface compatibility with vLLM and to
    allow for future improvements.

!!! info

    For diffusion pipelines, the stage config field `stage_args.[].runtime.max_batch_size` is 1 by
    default, and the input list is sliced into single-item requests before feeding into the diffusion
    pipeline. For models that do internally support batched inputs, you can
    [modify this configuration](../../../configuration/stage_configs.md) to let the model accept a
    longer batch of prompts.

### Negative Prompts

vLLM-Omni supports dictionary prompts for models that accept negative prompts:

```python
from vllm_omni.entrypoints.omni import Omni

if __name__ == "__main__":
    omni = Omni(model="Qwen/Qwen-Image")
    outputs = omni.generate([
        {
            "prompt": "a cup of coffee on a table",
            "negative_prompt": "low resolution"
        },
        {
            "prompt": "a toy dinosaur on a sandy beach",
            "negative_prompt": "cinematic, realistic"
        }
    ])
    for i, output in enumerate(outputs):
        output.request_output.images[0].save(f"{i}.jpg")
```

You can also pass a negative prompt via the CLI argument `--negative-prompt`:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model Qwen/Qwen-Image \
  --prompt "a cup of coffee on a table" \
  --negative-prompt "low resolution, blurry" \
  --output coffee.png
```

### Advanced Features

#### CFG Parallel

Set `--cfg-parallel-size 2` to enable CFG Parallel for faster inference on multi-GPU setups.
See more examples in the [cfg_parallel user guide](../../../docs/user_guide/parallelism/cfg_parallel.md#using-cfg-parallel).

#### LoRA

This example supports PEFT-compatible LoRA (Low-Rank Adaptation) adapters for diffusion models. Pass `--lora-path` to use a LoRA adapter and optionally `--lora-scale` (default `1.0`); omit it to use the base model only.

```bash
python text_to_image.py \
  --model Tongyi-MAI/Z-Image-Turbo \
  --prompt "A piece of cheesecake" \
  --lora-path /path/to/lora/ \
  --lora-scale 1.0 \
  --output output.png
```

LoRA adapters must be in PEFT format. A typical adapter directory structure:

```
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

## Web UI Demo

Launch the Gradio demo:

```bash
python gradio_demo.py --port 7862
```

Then open `http://localhost:7862/` in your local browser to interact with the web UI.
