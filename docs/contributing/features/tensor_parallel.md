# Support Tensor Parallel

This section describes how to add Tensor Parallel (TP) to a diffusion transformer model. We use the Z-Image transformer as the reference implementation.

---

## Table of Contents

- [Overview](#overview)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Reference Implementations](#reference-implementations)
- [Summary](#summary)

---

## Overview

### What is Tensor Parallel?

Tensor Parallel (TP) is a model parallelism technique that **shards model weights** across multiple GPUs. Each GPU holds only a portion of the model's parameters and computes only part of each layer's output.

Diffusion transformers contain large attention and MLP layers. We can use Tensor Parallel to shard the model dimension across multiple GPUs, allowing larger models to fit in memory while achieving near-linear speedup.

### Architecture

The Tensor Parallel implementation relies vLLM's Parallel Layers:

[vLLM Parallel Layers API Reference](https://docs.vllm.ai/en/latest/contributing/model/basic/?h=column#3-optional-implement-tensor-parallelism-and-quantization-support)

**Parallel Layer Types:**

| Layer Type | Purpose | Weight Partitioning |
|------------|---------|---------------------|
| `ColumnParallelLinear` | First FFN layer, separated QKV | Columns (output dimension) |
| `RowParallelLinear` | Second FFN layer, attention output | Rows (input dimension) |
| `QKVParallelLinear` | Multi-head/grouped-query attention QKV | Handles head replication automatically |
| `ReplicatedLinear` | Layers that shouldn't be sharded | No partitioning (replicated) |

---

## Step-by-Step Implementation


### Step 1: Identify Linear Layers

Find all `nn.Linear` layers in your transformer that need to be sharded.

**Key questions:**
- Which layers should be column parallel (weight split by columns)?
- Which layers should be row parallel (weight split by rows)?

### Step 2: Replace Linear Layers with Parallel Equivalents

Replace `nn.Linear` with parallel layers from `vllm.model_executor.layers.linear`.

**Example (MLP Block - Up-Down Pattern):**

```python
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # Column parallel: weight split by columns [hidden_dim/N, dim]
        self.w1 = ColumnParallelLinear(
            dim,
            hidden_dim,
            bias=False,
            return_bias=False,
        )
        self.act = nn.GELU()

        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,  # Input already sharded from w1
            return_bias=False,
        )

    def forward(self, x):
        # x: [batch, seq, dim] (replicated on all GPUs)
        # w1 outputs sharded [batch, seq, hidden_dim/N]
        x = self.w1(x)
        # act operates on sharded tensors (no communication)
        x = self.act(x)
        # w2 outputs full dim [batch, seq, dim] via all-reduce
        x = self.w2(x)
        return x
```

**Example (Attention - QKV-Out Pattern):**

```python
from vllm_omni.diffusion.attention.layer import Attention
class YourModelAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        self.head_dim = dim // num_heads

        # Column parallel: QKV weight split by columns
        # Each GPU gets num_heads/N heads
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
            return_bias=False,
        )

        # Row parallel: output weight split by rows
        self.to_out = RowParallelLinear(
            dim,
            dim,
            bias=False,
            input_is_parallel=True,  # Input sharded from attention
            return_bias=False,
        )

        self.attn = Attention(
            num_heads=self.to_qkv.num_heads, # Each GPU gets num_heads/N heads
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def forward(self, x):
        # x: [batch, seq, dim] (replicated)
        # to_qkv outputs sharded [batch, seq, (q+k+v) * head_dim/N]
        qkv = self.to_qkv(x)
        # Split into Q, K, V (each sharded on heads)
        q, k, v = qkv.split([...], dim=-1)
        # Attention computed independently on each GPU
        out = self.attn(q, k, v)
        # to_out all-reduces to full dim
        out = self.to_out(out)
        return out
```

**Key Points:**

- `ColumnParallelLinear` → `RowParallelLinear` is the standard pairing
- Set `input_is_parallel=True` on `RowParallelLinear` when input comes from `ColumnParallelLinear`
- Use `QKVParallelLinear` for attention projections (handles head replication automatically)

### Step 3: Validate TP Constraints

For correct TP operation, these dimensions **must be divisible** by `tensor_parallel_size`:

| Dimension | Reason | Example Error |
|-----------|--------|---------------|
| `num_heads` | Heads sharded by QKVParallelLinear | `num_heads=30, tp=4` ❌ (30 % 4 ≠ 0) |
| `num_kv_heads` | KV heads sharded by QKVParallelLinear | `num_kv_heads=30, tp=4` ❌ (30 % 4 ≠ 0) |

---

## Testing

After adding Tensor Parallel support, test with:

```python
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

parallel_config = DiffusionParallelConfig(tensor_parallel_size=2)
omni = Omni(model="your-model-name", parallel_config=parallel_config)

output = omni.generate(
    "a cup of coffee on the table",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

**Or via command line:**

```bash
cd examples/offline_inference/text_to_image
python text_to_image.py \
    --model Your-org/your-model \
    --prompt "a cup of coffee on the table" \
    --negative_prompt "ugly, unclear" \
    --cfg_scale 4.0 \
    --num_inference_steps 50 \
    --output "tp_enabled.png" \
    --tensor_parallel_size 2
```

**Verify:**

1. Check the `e2e_time_ms` in the log for speedup
2. Compare generated image quality with TP disabled
3. Verify memory usage is reduced proportionally
4. Record comparison results in your PR

---

## Troubleshooting

### Issue: TP not activating

**Symptoms:** Model runs on single GPU, no memory savings or speedup.

**Causes & Solutions:**

- **Still using `nn.Linear`:**

**Problem:** Linear layers not replaced with parallel equivalents.

**Solution:** Replace with parallel layers:
```python
# ❌ BAD
self.proj = nn.Linear(dim, dim)

# ✅ GOOD
self.proj = RowParallelLinear(dim, dim, input_is_parallel=True)
```

### Issue: Dimension mismatch errors

**Symptoms:** `RuntimeError: shape mismatch` during forward pass.

**Causes & Solutions:**

- **Missing `input_is_parallel=True`:**

**Problem:** RowParallelLinear expects sharded input but receives full tensor.

**Solution:** Set `input_is_parallel=True` when input comes from ColumnParallelLinear:
```python
# ✅ GOOD: Correct pairing
self.w1 = ColumnParallelLinear(dim, hidden_dim, return_bias=False,)
self.w2 = RowParallelLinear(
    hidden_dim,
    dim,
    input_is_parallel=True,  # Input sharded from w1
    return_bias=False,
)
```

- **Incorrect split dimensions:**

**Problem:** QKV split sizes don't match sharded dimensions.

**Solution:** Use `self.to_qkv.num_heads` (local heads per GPU):
```python
# ❌ BAD: Uses total heads
q_size = self.total_num_heads * self.head_dim

# ✅ GOOD: Uses local heads
q_size = self.to_qkv.num_heads * self.head_dim
```

---

## Reference Implementations

Complete examples in the codebase:

| Model | Path | Pattern | Notes |
|-------|------|---------|-------|
| **Z-Image** | `vllm_omni/diffusion/models/z_image/z_image_transformer.py` | Standard TP | Full implementation with validation |
| **FLUX** | `vllm_omni/diffusion/models/flux/flux_transformer.py` | Dual-stream | Image + text streams |
| **Qwen-Image** | `vllm_omni/diffusion/models/qwen_image/qwen_image_transformer.py` | Standard TP | With RoPE |
| **TP Tests** | `tests/e2e/offline_inference/test_zimage_tensor_parallel.py` | E2E testing | TP correctness and performance |
| **Constraint Tests** | `tests/diffusion/models/z_image/test_zimage_tp_constraints.py` | Unit testing | Validation logic |

---

## Summary

Adding Tensor Parallel support to a transformer:

1. ✅ **Identify linear layers** - Which layers should be sharded?
2. ✅ **Replace with parallel layers** - Use QKVParallelLinear, ColumnParallelLinear, RowParallelLinear
3. ✅ **Validate TP constraints** - Ensure dimensions divisible by TP size
4. ✅ **Test** - Verify with `tensor_parallel_size=N`, check memory, speed, and quality
