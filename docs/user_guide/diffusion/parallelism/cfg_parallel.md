# CFG-Parallel Guide

## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Example Script](#example-script)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Summary](#summary)

---

## Overview

CFG-Parallel accelerates diffusion models by distributing positive and negative classifier-free guidance (CFG) passes across different GPUs, providing ~1.8x speedup when CFG is enabled. It's ideal for image editing tasks that require guidance scales greater than 1.0.

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).

Distilled Cosmos3 checkpoints, including `nvidia/Cosmos3-Super-Image2Video-4Step`,
run without CFG and reject `--cfg-parallel-size` greater than 1 at startup.
Use `--ulysses-degree 2` for two-GPU inference. These checkpoints always use
`guidance_scale=1.0`; other explicitly requested values produce a warning, and
`negative_prompt` does not affect generation.

---

## Quick Start

### Basic Usage

Simplest working example:

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from PIL import Image

omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    parallel_config=DiffusionParallelConfig(cfg_parallel_size=2),  # Enable CFG-Parallel
)

input_image = Image.open("input.png").convert("RGB")
outputs = omni.generate(
    {
        "prompt": "turn this cat to a dog",
        "negative_prompt": "low quality, blurry",
        "multi_modal_data": {"image": input_image},
    },
    OmniDiffusionSamplingParams(
        true_cfg_scale=4.0,
        num_inference_steps=50,
    ),
)
```

---

## Example Script

### Offline Inference

Use python script under `examples/offline_inference/image_to_image/image_edit.py`:

```bash
cd examples/offline_inference/image_to_image/
python image_edit.py \
  --model "Qwen/Qwen-Image-Edit" \
  --image "input.png" \
  --prompt "turn this cat to a dog" \
  --negative-prompt "low quality, blurry" \
  --cfg-scale 4.0 \
  --output "edited_image.png" \
  --cfg-parallel-size 2
```

### Online Serving

Enable CFG-Parallel in online serving:

```bash
# Default configuration
vllm serve Qwen/Qwen-Image-Edit --omni --port 8091 --cfg-parallel-size 2

```

---

## Configuration Parameters

In `DiffusionParallelConfig`

| Parameter           | Type | Default | Description                                                                                                           |
| ------------------- | ---- | ------- | --------------------------------------------------------------------------------------------------------------------- |
| `cfg_parallel_size` | int  | 1       | Number of GPUs for CFG parallelism. Set to 2 to enable CFG-Parallel (rank 0 for positive, rank 1 for negative branch) |

!!! info
    Most models support `cfg_parallel_size=2` (positive branch on rank 0, negative branch on rank 1). **Bagel** and **Boogu-Image-Edit double guidance** also support `cfg_parallel_size=3` for full three-way branch execution. Boogu double guidance additionally supports size 2 via round-robin assignment of its three branches.

### CFG batching on one CFG rank

Without CFG-Parallel (`cfg_parallel_size=1`), some pipelines run the positive and negative branch as one denoiser
call on the two branches concatenated along the batch dimension, instead of two calls. This halves the denoiser
calls per step, which helps most where a step is launch-bound (small resolutions, eager mode). It is on by default
(`enable_cfg_batching`); disable it with `--no-enable-cfg-batching` or `Omni(..., enable_cfg_batching=False)`.

- Pipelines opt in with `supports_cfg_batching = True`: SDXL, SD3, FLUX.1 (true CFG) and FLUX.1-Kontext,
  FLUX.2 and FLUX.2-Klein, Wan2.2 T2V/TI2V/I2V/VACE.
- It is not used with CFG-Parallel, pipeline parallelism, or any `cache_backend` (TeaCache and friends keep one state
  per branch). If the branch arguments differ in shape (for example different prompt lengths), that step runs the
  branches sequentially and a warning is logged once.
- Batched and sequential outputs are not bit-identical in bf16, since GEMMs run at twice the batch size.

---

## Best Practices

### When to Use

**Good for:**

- Tasks requiring classifier-free guidance
- Multi-GPU setups (at least 2 GPUs available)
- Combining with other parallelism methods (sequence/tensor parallel)

**Not for:**

- Single GPU setups
- Models that don't support CFG-Parallel (check [supported models](../../diffusion_features.md#supported-models))
- Workloads without negative prompts or classifier-free guidance
- Very short inference runs (< 10 steps) where parallelism overhead may outweigh benefits

### Expected Performance

| Configuration         | Speedup  | Quality        | Use Case                  |
| --------------------- | -------- | -------------- | ------------------------- |
| CFG-Parallel (2 GPUs) | 1.5~1.8x | No degradation | Large model, VRAM limited |

---

## Troubleshooting

### Common Issue 1: No Speedup with CFG-Parallel

**Symptoms**: CFG-Parallel enabled but no performance improvement

**Solutions**:

1. **Ensure CFG scale is set correctly:**

```python
# Bad: No CFG effect
sampling_params = OmniDiffusionSamplingParams(num_inference_steps=50)

# Good: CFG-Parallel will work
sampling_params = OmniDiffusionSamplingParams(
    num_inference_steps=50,
    true_cfg_scale=4.0  # Must be > 1.0
)
```

1. **Add negative prompt:**

```python
outputs = omni.generate(
    {
        "prompt": "beautiful landscape",
        "negative_prompt": "low quality, blurry",  # Required for best results
        "multi_modal_data": {"image": input_image}
    },
    sampling_params
)
```

1. **Check model support:**
   - Verify your model in [supported models](../../diffusion_features.md#supported-models)
   - Some models don't support CFG-Parallel

---

## Summary

1. ✅ **Enable CFG-Parallel** - Set `cfg_parallel_size=2` in `DiffusionParallelConfig` to get speedup when using CFG
2. ✅ **Set CFG Scale** - Ensure `true_cfg_scale > 1.0` in `OmniDiffusionSamplingParams` for CFG-Parallel to take effect
3. ✅ **Check Model Support** - Verify your model supports CFG-Parallel in [supported models](../../diffusion_features.md#supported-models)
