# LoRA (Low-Rank Adaptation) Guide

LoRA (Low-Rank Adaptation) enables fine-tuning diffusion models by adding trainable low-rank matrices to existing model weights. vLLM-Omni currently supports PEFT-style LoRA adapters, allowing you to customize model behavior without modifying the base model weights.

## Overview

LoRA adapters are lightweight, model-specific fine-tuning weights that can be dynamically loaded and applied to diffusion models. vLLM-Omni uses a unified LoRA handling mechanism similar to vLLM with LRU cache management.

## LoRA Adapter Format

LoRA adapters must be in **PEFT (Parameter-Efficient Fine-Tuning)** format. A typical LoRA adapter directory structure:

```
lora_adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

The `adapter_config.json` file contains metadata about the LoRA adapter, including:
- `r`: LoRA rank
- `lora_alpha`: LoRA alpha scaling factor
- `target_modules`: List of module names to apply LoRA to

## Quick Start

### Offline Inference

#### Pre-loaded LoRA

Load a LoRA adapter at initialization. This adapter is pre-loaded into the cache and can be activated by requests:

```python
from vllm_omni import Omni
from vllm_omni.lora.request import LoRARequest

lora_path="/path/to/lora_adapter"

omni = Omni(
    model="stabilityai/stable-diffusion-3.5-medium",
    lora_path=lora_path
)

lora_request = LoRARequest(
    lora_name="preloaded",
    lora_int_id=1,
    lora_path=lora_path
)

outputs = omni.generate(
    prompt="A piece of cheesecake",
    lora_request=lora_request,
    lora_scale=2.0, # optional arg, default 1.0
)
```

!!! note "Server-side Path Requirement"
    The LoRA adapter path (`local_path`) must be readable on the **server** machine. If your client and server are on different machines, ensure the LoRA adapter is accessible via a shared mount or copied to the server.

## Wan2.2 LightX2V Offline Assembly

This workflow is LoRA-adjacent: it uses external LightX2V conversion plus
`Wan2.2-Distill-Loras` to bake converted Wan2.2 I2V checkpoints into a local
Diffusers directory, instead of loading LoRA adapters at runtime.

### Required assets

- Base model: `Wan-AI/Wan2.2-I2V-A14B`
- Diffusers skeleton: `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
- Optional external converter from the LightX2V project (not shipped in this repository)
- Optional LoRA weights: `lightx2v/Wan2.2-Distill-Loras`

### Step 1: Optional - convert high/low-noise DiT weights with LightX2V

Install or clone LightX2V from the upstream repository
(`https://github.com/ModelTC/LightX2V`). After cloning, the converter used
below is available at `<lightx2v_root>/tools/convert/converter.py`.

```bash
python /path/to/lightx2v/tools/convert/converter.py \
  --source /path/to/Wan2.2-I2V-A14B/high_noise_model \
  --output /tmp/wan22_lightx2v/high_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file

python /path/to/lightx2v/tools/convert/converter.py \
  --source /path/to/Wan2.2-I2V-A14B/low_noise_model \
  --output /tmp/wan22_lightx2v/low_noise_out \
  --output_ext .safetensors \
  --output_name diffusion_pytorch_model \
  --model_type wan_dit \
  --direction forward \
  --lora_path /path/to/wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step_1022.safetensors \
  --lora_key_convert auto \
  --single_file
```

If you are not using LightX2V, skip this step and either keep the original
Diffusers weights from the skeleton or point Step 2 at any other converted
`transformer/` and `transformer_2/` checkpoints.

### Step 2: Assemble a final Diffusers-style directory

```bash
python tools/wan22/assemble_wan22_i2v_diffusers.py \
  --diffusers-skeleton /path/to/Wan2.2-I2V-A14B-Diffusers \
  --transformer-weight /tmp/wan22_lightx2v/high_noise_out \
  --transformer-2-weight /tmp/wan22_lightx2v/low_noise_out \
  --output-dir /path/to/Wan2.2-I2V-A14B-Custom-Diffusers \
  --asset-mode symlink \
  --overwrite
```

`--transformer-weight` and `--transformer-2-weight` are optional. If you omit
them, the tool keeps the original weights from the Diffusers skeleton.

### Step 3: Run offline inference

```bash
python examples/offline_inference/image_to_video/image_to_video.py \
  --model /path/to/Wan2.2-I2V-A14B-Custom-Diffusers \
  --image /path/to/input.jpg \
  --prompt "A cat playing with yarn" \
  --num-frames 81 \
  --num-inference-steps 4 \
  --tensor-parallel-size 4 \
  --height 480 \
  --width 832 \
  --flow-shift 12 \
  --sample-solver euler \
  --guidance-scale 1.0 \
  --guidance-scale-high 1.0 \
  --boundary-ratio 0.875
```

Notes:

- This route avoids runtime LoRA loading changes in vLLM-Omni when you choose to bake converted weights into a local Diffusers directory.
- Output quality and speed depend on the replacement checkpoints and sampling params you choose.


## See Also

- [Text-to-Image Offline Example](../examples/offline_inference/text_to_image.md#lora) - Complete offline LoRA example
- [Text-to-Image Online Example](../examples/online_serving/text_to_image.md#lora) - Complete online LoRA example
