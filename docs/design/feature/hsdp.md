# HSDP

This section describes how to add HSDP (Hybrid Sharded Data Parallel) support to a diffusion transformer model. We use the Wan2.2 transformer as the reference implementation.

---

## Table of Contents

- [Overview](#overview)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Testing](#testing)
- [Reference Implementations](#reference-implementations)

---

## Overview

### What is HSDP?

HSDP (Hybrid Sharded Data Parallel) is a memory optimization technique that **shards model weights** across multiple GPUs using PyTorch's FSDP2. Unlike Tensor Parallelism which splits computation, HSDP:

- Shards weights across GPUs to reduce per-GPU memory usage
- Gathers weights on-demand during forward passes
- Can work standalone or combined with other parallelism (e.g., Sequence Parallel)

This enables inference of large models (e.g., Wan2.2 14B) on GPUs with limited memory.

**Important constraints:**
- HSDP cannot be used with Tensor Parallelism
- For standalone HSDP (no other parallelism), `hsdp_shard_size` must be specified explicitly

### Architecture

HSDP implementation relies on:

1. **`_hsdp_shard_conditions`**: Model attribute specifying which modules to shard
2. **`apply_hsdp_to_model`**: Function that applies FSDP2 sharding
3. **`HSDPInferenceConfig`**: Runtime configuration for HSDP

---

## Step-by-Step Implementation

### Step 1: Identify Modules to Shard

Determine which modules in your transformer should be sharded. Typically, these are:

- Transformer blocks (e.g., `blocks.0`, `blocks.1`, ...)
- Large submodules with significant weight memory

**Key questions:**
- Which modules have the largest weights?
- Which modules are repeated (like transformer blocks)?

### Step 2: Define Shard Conditions

Add `_hsdp_shard_conditions` to your model class. This is a list of functions that return `True` for modules that should be sharded.

**Example (Transformer Blocks):**

```python
class MyTransformerModel(nn.Module):

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        """Match transformer blocks for HSDP sharding.

        Args:
            name: Module name from named_modules() (e.g., "blocks.0", "blocks.0.attn")
            module: The module instance

        Returns:
            True if this module should be sharded
        """
        return "blocks" in name and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]
```

**Multiple Conditions Example:**

```python
class MyModel(nn.Module):

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return "blocks" in name and name.split(".")[-1].isdigit()

    @staticmethod
    def _is_moe_expert(name: str, module) -> bool:
        # Also shard MoE expert layers
        return "experts" in name and name.split(".")[-1].isdigit()

    # Module is sharded if ANY condition returns True
    _hsdp_shard_conditions = [_is_transformer_block, _is_moe_expert]
```

---

## Testing

After adding HSDP support, test with:

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

parallel_config = DiffusionParallelConfig(
    use_hsdp=True,
    hsdp_shard_size=8,  # Shard across 8 GPUs
)
omni = Omni(model="your-model-name", parallel_config=parallel_config)

output = omni.generate(
    "a cup of coffee on the table",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

**Or via command line:**

```bash
vllm serve Your-org/your-model --omni --port 8091 --use-hsdp
```

**Verify:**

1. Check logs for "HSDP Inference: replicate_size=..., shard_size=..."
2. Check logs for "Sharded N modules + root"
3. Verify memory usage is reduced proportionally
4. Compare generated output quality with HSDP disabled

---

## Reference Implementations

Complete examples in the codebase:

| Model | Path | Notes |
|-------|------|-------|
| **Wan2.2** | `vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py` | Reference implementation |
| **HSDP Core** | `vllm_omni/diffusion/distributed/hsdp.py` | `apply_hsdp_to_model`, `shard_model` |
| **HSDP Tests** | `tests/diffusion/distributed/test_hsdp.py` | Unit tests |

---
