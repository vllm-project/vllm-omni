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


## See Also

- [Text-to-Image Offline Example](../examples/offline_inference/text_to_image.md#lora) - Complete offline LoRA example
- [Text-to-Image Online Example](../examples/online_serving/text_to_image.md#lora) - Complete online LoRA example
