# Support Sequence Parallel

This section describes how to add Sequence Parallel (SP) to a diffusion transformer model. We use the Qwen-Image transformer and Wan2.2 transformer as reference implementations.

---

## Table of Contents

- [Overview](#overview)
- [Approach 1: Non-Intrusive `_sp_plan` (Recommended)](#approach-1-non-intrusive-_sp_plan-recommended)
- [Approach 2: Intrusive Modification (For Complex Cases)](#approach-2-intrusive-modification-for-complex-cases)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Reference Implementations](#reference-implementations)
- [Summary](#summary)

---

## Overview


### What is Sequence Parallel?

**Terminology Note:** Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP) in the [diffusers library](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/_modeling_parallel.py). We use "Sequence Parallelism" to align with vLLM-Omni's terminology.

Diffusion transformers process long sequences of image patches or video frames. For high-resolution generation, these sequences can become very large. Enabling SP allows each GPU to process only a portion of the sequence, with attention mechanisms (Ulysses/Ring) handling cross-GPU communication transparently.

### Architecture

The major APIs for Sequence Parallel:

```python
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,   # For sharding (splitting) tensors
    SequenceParallelOutput,  # For gathering tensors
)
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard, sp_gather
```

| Method/Class | Purpose | Behavior |
|--------------|---------|----------|
| `SequenceParallelInput` | Declare input sharding in `_sp_plan` | Auto-shards tensors at module input |
| `SequenceParallelOutput` | Declare output gathering in `_sp_plan` | Auto-gathers tensors at module output |
| `sp_shard()` | Manual tensor sharding | Splits tensor across SP workers |
| `sp_gather()` | Manual tensor gathering | Gathers sharded tensors from all workers |

---

## Approach 1: Non-Intrusive `_sp_plan` (Recommended)

The `_sp_plan` mechanism allows SP **without modifying `forward()` logic**. The framework automatically registers hooks to shard inputs and gather outputs at module boundaries.

**When to use:**
- Standard transformer architectures
- Tensor operations happen at `nn.Module` boundaries
- Predictable sharding/gathering patterns

**How it works:**
1. Declare `_sp_plan` dict in your transformer class
2. Framework automatically applies hooks when `sequence_parallel_size > 1`
3. Hooks shard/gather tensors at specified module boundaries
4. Attention layers handle cross-GPU communication internally

```python
class StandardTransformer(nn.Module):
    _sp_plan = {
        # Shard hidden_states at first transformer block input
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        # Gather at final output projection
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

`StandardTransformer` has a transformer blocks list `self.blocks = nn.ModuleList([...])`, and a projection output layer `self.proj_out`. The `_sp_plan` above defines that when SP is enabled, sharding the input tensor to the first transformer block, and gathering the sharded tensor at the final output projection layer.

**Requirements:**
- Tensor operations that need sharding/gathering must happen at **`nn.Module` boundaries**
- Inline Python operations (e.g., `torch.cat`, `pad_sequence`) **cannot be hooked**

**Solution for inline operations:** Extract into a submodule (see Step 2 below).

### Step 1: Understand Module Boundaries

Identify where tensors need to be sharded or gathered in your model's forward pass:

```python
class MyTransformer(nn.Module):
    def __init__(self):
        self.patch_embed = PatchEmbed()      # ← Boundary 1
        self.pos_embed = RoPE()              # ← Boundary 2
        self.blocks = nn.ModuleList([...])   # ← Boundary 3
        self.norm_out = LayerNorm()
        self.proj_out = Linear()             # ← Boundary 4

    def forward(self, x):
        x = self.patch_embed(x)              # ← Shard before this?
        pos = self.pos_embed(x)              # ← Shard RoPE outputs?
        for block in self.blocks:
            x = block(x, pos)                # ← Blocks process sharded x
        x = self.norm_out(x)
        output = self.proj_out(x)            # ← Gather after this?
        return output
```

### Step 2: Handle Inline Operations

If your `forward()` contains inline tensor operations, **extract them into submodules**.

**Example: Z-Image concatenates image + text features inline**

```python
# ❌ BAD: Inline operation - hooks cannot intercept
class ZImageTransformer(nn.Module):
    def forward(self, x, cap_feats):
        # This concatenation happens inline - _sp_plan can't shard it!
        unified = torch.cat([x, cap_feats], dim=1)

        for layer in self.layers:
            unified = layer(unified)

        return unified

# ✅ GOOD: Extract into submodule
class UnifiedPrepare(nn.Module):
    """Submodule to concatenate image and text features."""
    def forward(self, x, cap_feats):
        return torch.cat([x, cap_feats], dim=1)

class ZImageTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.unified_prepare = UnifiedPrepare()  # Now a module!
        self.layers = nn.ModuleList([...])

    def forward(self, x, cap_feats):
        # Now _sp_plan can shard the output of unified_prepare!
        unified = self.unified_prepare(x, cap_feats)

        for layer in self.layers:
            unified = layer(unified)

        return unified
```

**Other common cases:**
- `pad_sequence()` → `PadSequenceModule`
- `torch.cat()` → `ConcatModule`
- `tensor.reshape()` → `ReshapeModule`
- Complex preprocessing → `PreprocessModule`

### Step 3: Write `_sp_plan` for Your Model

Create a class-level `_sp_plan` dictionary specifying where to shard/gather tensors.

Typically, there are two patterns for diffusion models:

**Pattern 1: Shard at first block, gather at output projection**

Most common pattern for standard transformers:

```python
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,   # For sharding (splitting) tensors
    SequenceParallelOutput,  # For gathering tensors
)
class StandardTransformer(nn.Module):
    _sp_plan = {
        # Shard hidden_states at first transformer block input
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        # Gather at final output projection
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

**Pattern 2: Shard RoPE embeddings separately**

When RoPE is computed in a separate module:

```python
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,   # For sharding (splitting) tensors
    SequenceParallelOutput,  # For gathering tensors
)
class TransformerWithRoPE(nn.Module):
    _sp_plan = {
        # Shard RoPE module OUTPUTS (returns tuple of cos, sin)
        "rope": {
            0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # cos
            1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),  # sin
        },
        # Shard transformer block INPUT
        "blocks.0": {
            "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3),
        },
        # Gather at output
        "proj_out": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }
```

### API Reference

**SequenceParallelInput Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `split_dim` | int | Dimension to split (usually `1` for sequence) |
| `expected_dims` | int \| None | Expected tensor rank for validation (optional) |
| `split_output` | bool | `False`: shard **input** params; `True`: shard **output** tensors |
| `auto_pad` | bool | Auto-pad if sequence not divisible by world_size (default: `False`) |

**SequenceParallelOutput Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `gather_dim` | int | Dimension to gather (usually `1` for sequence) |
| `expected_dims` | int \| None | Expected tensor rank for validation (optional) |

**Module Naming Conventions:**

| Key | Meaning | Python equivalent |
|-----|---------|-------------------|
| `""` | Root model | `model` |
| `"blocks.0"` | First element of ModuleList | `model.blocks[0]` |
| `"blocks.*"` | All elements of ModuleList | `for b in model.blocks` |
| `"rope"` | Named submodule | `model.rope` |
| `"outputs.main"` | ModuleDict entry | `model.outputs["main"]` |

**Dictionary Value Types:**

| Key type | `split_output` | Description |
|----------|----------------|-------------|
| `"param_name"` (str) | `False` | Shard **input parameter** by name |
| `0`, `1`, ... (int) | `True` | Shard **output tuple** by index |

---

## Approach 2: Intrusive Modification (For Complex Cases)

For models with dynamic sharding logic that cannot be expressed via `_sp_plan`, manually insert shard/gather calls.

**When to use:**
- Dynamic/conditional sharding logic
- Complex tensor manipulations that can't be encapsulated
- Temporary workaround during development

```python
from vllm_omni.diffusion.distributed.sp_sharding import sp_shard, sp_gather

def forward(self, hidden_states, ...):
    if self.parallel_config.sequence_parallel_size > 1:
        hidden_states = sp_shard(hidden_states, dim=1)

    # ... computation ...

    if self.parallel_config.sequence_parallel_size > 1:
        output = sp_gather(output, dim=1)

    return output
```

---

## Testing

After implementing Sequence Parallel support, thoroughly test your implementation to ensure correctness and performance across different configurations.

**Test Different `sp_size`:**

Test your model with various sequence parallel world sizes to verify correctness and identify optimal configurations:

```bash
cd examples/offline_inference/text_to_image
python text_to_image.py \
    --model Your-org/your-model \
    --prompt "a cup of coffee on the table" \
    --num_inference_steps 50 \
    --ulysses_degree 2 \
    --ring_degree 2 \
    --output sp_test_image_ulysses=2_ring=2.png
```

**Verify:**

1. **Correctness:** Output should be identical across all `sp_size` values
2. **Speed:** Throughput should remain stable or improve (especially for large sequences)
3. **Logs:** Check for any shape mismatch or communication errors

**Test with Tensor Parallel:**

Sequence Parallel can be combined with other parallelism strategies:

```bash
cd examples/offline_inference/text_to_image
python text_to_image.py \
    --model Your-org/your-model \
    --prompt "a cup of coffee on the table" \
    --num_inference_steps 50 \
    --ulysses_degree 2 \
    --tensor_parallel_size 2 \
    --output sp_test_image_ulysses=2_tp=2.png
```

---

## Troubleshooting

### Issue: Shape mismatch errors

**Symptoms:** `RuntimeError: shape mismatch` during forward pass.

**Causes & Solutions:**

- **RoPE dimension mismatch:**

**Problem:** RoPE embeddings not sharded, but hidden_states is sharded.

**Solution:** Shard RoPE outputs in `_sp_plan`:
```python
_sp_plan = {
    "rope": {
        0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
        1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
    },
    ...
}
```

- **Sequence Length not divisible by sp_size:**

**Problem:** `SequenceParallelInput(auto_pad=False)` - auto_pad should be True to enable automatic sequence padding.

**Solution:** In `SequenceParallelInput`, set `auto_pad=True`:
```python
"blocks.0": {
    "hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True)
}
```

### Issue: Inline operations not sharded

**Symptoms:** Some tensors remain full-sized, not sharded.

**Causes & Solutions:**

- **Operations happen inline in `forward()`, not at module boundaries:**

**Problem:**
```python
def forward(self, x, cap):
    unified = torch.cat([x, cap], dim=1)  # ← Inline operation!
    # _sp_plan can't hook this
```

**Solution:** Extract into submodule:
```python
class ConcatModule(nn.Module):
    def forward(self, x, cap):
        return torch.cat([x, cap], dim=1)

class MyModel(nn.Module):
    _sp_plan = {
        "concat": {
            0: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        ...
    }
    def __init__(self):
        self.concat = ConcatModule()  # Now hookable!

    def forward(self, x, cap):
        unified = self.concat(x, cap)  # ← Can be sharded via _sp_plan
```

---

## Reference Implementations

Complete examples in the codebase:

| Model | Path | Pattern | Notes |
|-------|------|---------|-------|
| **Qwen-Image** | `vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py` | Dual-stream + preprocessing | auto_pad, separate RoPE |
| **Wan2.2** | `vllm_omni/diffusion/models/wan2_2/wan2_2_transformer.py` | Dual-Transformer + RoPE | Video transformer |
| **Z-Image** | `vllm_omni/diffusion/models/z_image/z_image_transformer.py` | Unified sequence | Concatenated input |
| **SP Plan Types** | `vllm_omni/diffusion/distributed/sp_plan.py` | Type definitions | SequenceParallelInput/Output |
| **Hook Implementation** | `vllm_omni/diffusion/hooks/sequence_parallel.py` | Hook mechanics | How hooks work |
| **Tests** | `tests/diffusion/distributed/test_sp_plan_hooks.py` | Test examples | Validation patterns |

---

## Summary

Adding Sequence Parallel support to a transformer:

1. ✅ **Choose approach** - Use `_sp_plan` for standard cases, intrusive modification for complex cases
2. ✅ **Identify sharding boundaries** - Where should tensors be split/gathered?
3. ✅ **Extract inline operations** - Move `torch.cat`, `pad_sequence`, etc. to submodules
4. ✅ **Define `_sp_plan`** - Declare shard/gather points as class attribute
5. ✅ **Use `auto_pad` for variable lengths** - Support non-uniform sequences
6. ✅ **Test** - Verify with different `ulysses_degree` and `ring_degree` combinations
