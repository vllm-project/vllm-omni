# Custom Pipeline Extension Guide

Transformer already support Custom Pipeline via
https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/custom_pipeline_overview.md

This guide demonstrates how to use the newly added features for extending vLLM-Omni's diffusion pipeline with custom functionality.

## Overview

Three main features enable custom pipeline extension:

1. **`WorkerWrapperBase`**: A wrapper class that enables dynamic worker extension with custom functionality
2. **`load_format`**: A parameter that controls how diffusion models are loaded, including support for custom pipelines
3. **`CustomPipelineWorkerExtension`**: An extension class that enables pipeline re-initialization with custom implementations

## Features

### WorkerWrapperBase

`WorkerWrapperBase` is a wrapper class that creates `DiffusionWorker` instances with optional extension support. It enables dynamic inheritance, allowing you to add custom methods and functionality to workers without modifying the base worker class.

**Key capabilities:**
- Dynamic worker class extension via `worker_extension_cls`
- Support for custom pipeline initialization via `custom_pipeline_args`
- Method delegation to underlying worker
- Attribute access forwarding

**Location:** `vllm_omni/diffusion/worker/diffusion_worker.py`

### load_format Parameter

The `load_format` parameter controls how diffusion models are loaded. It supports the following values:

- **`"default"`**: Standard model loading using the model registry (default behavior)
- **`"custom_pipeline"`**: Load a custom pipeline class specified by `custom_pipeline_name`
- **`"dummy"`**: Skip model loading (useful for testing or when pipeline will be initialized separately)

**Location:** `vllm_omni/diffusion/model_loader/diffusers_loader.py`

### CustomPipelineWorkerExtension

`CustomPipelineWorkerExtension` is a mixin class that extends `DiffusionWorker` with the ability to re-initialize the pipeline with a custom implementation.

**Key method:**
- `re_init_pipeline(custom_pipeline_args)`: Re-initializes the pipeline with custom arguments, properly cleaning up the old pipeline first

**Location:** `vllm_omni/diffusion/worker/diffusion_worker.py`

## Usage Example

### Step 1: Create a Custom Pipeline

Create a custom pipeline class that extends an existing pipeline. In this example, we extend `QwenImageEditPipeline` to add trajectory tracking:

```python
# custom_pipeline.py
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import QwenImageEditPipeline
import torch

class CustomPipeline(QwenImageEditPipeline):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)

    def forward(self, req, prompt=None, negative_prompt=None, **kwargs):
        # Call parent's forward to get normal output
        output = super().forward(req=req, prompt=prompt, negative_prompt=negative_prompt, **kwargs)

        # Add custom trajectory data
        actual_num_steps = req.sampling_params.num_inference_steps or kwargs.get('num_inference_steps', 50)
        output.trajectory_timesteps = torch.linspace(1000, 0, actual_num_steps, dtype=torch.float32)
        output.trajectory_latents = torch.randn(actual_num_steps, 1, 16, 64, 64, dtype=torch.float32)

        return output
```

### Step 2: Use the Custom Pipeline with Omni

Initialize the `Omni` engine with custom pipeline configuration:

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# Initialize with custom pipeline
omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    diffusion_load_format="dummy",  # Skip initial loading
    custom_pipeline_args={
        "pipeline_class": "custom_pipeline.CustomPipeline"
    },
)

# Generate with the custom pipeline
outputs = omni.generate(
    ...
)

# Access custom trajectory data
output = outputs[0].request_output[0]
print(f"Trajectory timesteps shape: {output.metrics['trajectory_timesteps'].shape}")
print(f"Trajectory latents shape: {output.latents.shape}")
```

### Step 3: Run the Example

The example provided in this directory demonstrates the complete workflow:

```bash
cd examples/offline_inference/custom_pipeline/image_to_image

# Run with custom pipeline
python image_edit.py \
    --model Qwen/Qwen-Image-Edit-2511 \
    --image cherry_blossom.jpg \
    --prompt "Let this mascot dance under the moon, surrounded by floating stars" \
    --output output_image_edit.png \
    --num_inference_steps 10
```

## Advanced Usage

### Custom Worker Extension

You can create custom worker extensions to add new methods beyond pipeline re-initialization:

```python
from typing import Any
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

class MyCustomExtension(CustomPipelineWorkerExtension):
    def custom_method(self):
        """Your custom worker method."""
        return "custom_result"

    def another_method(self, data: Any):
        """Another custom method."""
        # Access worker internals via self
        return self.model_runner.some_operation(data)

omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    diffusion_load_format="dummy",
    custom_pipeline_args={
        "pipeline_class": "custom_pipeline.CustomPipeline"
    },
    worker_extension_cls=MyCustomExtension,
    # Note: worker_extension_cls is an internal parameter
    # CustomPipelineWorkerExtension will automatically init pipeline when custom_pipeline_args is provided
)
```
