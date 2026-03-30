# Expert Parallelism Guide


## Table of Content

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Configuration Parameters](#configuration-parameters)
- [Best Practices](#best-practices)
- [Summary](#summary)

---

## Overview

Unlike Tensor Parallelism which shards every layer's weights, Expert Parallelism (EP) only shards the MoE expert MLP blocks. This significantly reduces the memory footprint of MoE models (e.g., HunyuanImage3.0) while maintaining constant dense-equivalent compute efficiency.

During the forward pass, a gating mechanism routes tokens to their designated experts, requiring all-to-all communication to dispatch tokens to the correct ranks and combine results.

See supported models list in [Supported Models](../../diffusion_features.md#supported-models).

!!! note "EP Size Constraint"
    The effective EP size equals `tp × sp × cfg × dp`. At least one of TP/SP/CFG/DP must be set when EP is enabled.

---

## Quick Start

### Basic Usage

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.diffusion.data import DiffusionParallelConfig

omni = Omni(
    model="tencent/HunyuanImage-3.0",
    parallel_config=DiffusionParallelConfig(
        tensor_parallel_size=8,
        enable_expert_parallel=True,
    ),
)

outputs = omni.generate(
    "A brown and white dog is running on the grass",
    OmniDiffusionSamplingParams(
        num_inference_steps=50,
        width=1024,
        height=1024,
    ),
)
```

---

## Configuration Parameters

In `DiffusionParallelConfig`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_expert_parallel` | bool | False | Enable Expert Parallelism for MoE models |

EP size is derived automatically as `tp × sp × cfg × dp` — configure at least one of those to set the EP degree.

---

## Best Practices

### When to Use

**Good for:**

- MoE models (e.g., HunyuanImage3.0) with numbers of experts
- Memory-constrained multi-GPU setups where only expert blocks need sharding

**Not for:**

- Dense models (no MoE layers) — EP has no effect
- Single GPU setups

---

## Summary

1. ✅ **Enable EP** - Set `enable_expert_parallel=True` in `DiffusionParallelConfig` for MoE models
2. ✅ **Set parallelism degree** - At least one of `tensor_parallel_size` / `ulysses_degree` / `cfg_parallel_size` must be > 1 to define the EP size
