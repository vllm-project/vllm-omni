## Your current environment

<details>
<summary>The output of <code>python collect_env.py</code></summary>

```text
The full `collect_env.py` output was not saved in the AMD container. The ROCm verification runner recorded these values:
PyTorch: 2.11.0+gitd0c8b1f
HIP: 7.2.53211
vLLM: 0.27.0
vLLM-Omni commit: e20bcd5231f2f8ffb10fe6e337c303305cde9118
ROCm platform: True
Device count: 2
GPU architecture: gfx942
```

</details>

## Your code version

<details>
<summary>The commit id or version of vLLM</summary>

```text
v0.27.0
```
</details>

<details>
<summary>The commit id or version of vLLM-Omni</summary>

```text
The bug is present on current upstream main at 678aa25864334286d778fb3853b3c34f4b2ef0a6.
The proposed fix is currently squashed at ef5324668a75400cbe2cb5cd826582eed9f3daf3.
The AMD validation was run before the history-only rebase and squash at e20bcd5231f2f8ffb10fe6e337c303305cde9118. Every file changed by the fix has identical contents at both commits.
```
</details>

## 🐛 Describe the bug

Online diffusion INT8 has three related problems on current main.

First, `DiffusionInt8Config.get_quant_method()` accepts CUDA and NPU but rejects ROCm, even though vLLM 0.27.0 provides the ROCm Triton and AITER INT8 linear kernels through `init_int8_linear_kernel()`.

Second, `Int8OnlineLinearMethod.process_weights_after_loading()` calls `ops.scaled_int8_quant(layer.weight, scale=None)` independently on every tensor-parallel rank. For a row-parallel layer, each rank owns only part of the input dimension. The kernel therefore computes a different per-output-row scale from each local shard. TP2 then represents a different quantized weight and can produce different output from TP1 for reasons beyond the expected parallel reduction order.

The following minimal example simulates two input shards on one ROCm GPU. It does not need a model checkpoint.

```python
import torch
from vllm import _custom_ops as ops

weight = torch.tensor(
    [[1, -2, 8, -64], [2, -3, 7, -56]],
    dtype=torch.bfloat16,
    device="cuda",
)

full_qweight, full_scale, _ = ops.scaled_int8_quant(weight, scale=None)
rank0_qweight, rank0_scale, _ = ops.scaled_int8_quant(
    weight[:, :2].contiguous(),
    scale=None,
)

print("full scale:", full_scale.cpu())
print("rank 0 scale:", rank0_scale.cpu())
print("full rank 0 shard:", full_qweight[:, :2].cpu())
print("rank 0 local quantization:", rank0_qweight.cpu())

assert torch.equal(rank0_scale, full_scale)
assert torch.equal(rank0_qweight, full_qweight[:, :2])
```

Observed result:

```text
full scale: tensor([[0.5039], [0.4409]])
rank 0 scale: tensor([[0.0157], [0.0236]])
full rank 0 shard: tensor([[ 2, -4], [ 5, -7]], dtype=torch.int8)
rank 0 local quantization: tensor([[ 64, -127], [ 85, -127]], dtype=torch.int8)
AssertionError
```

Expected result:

Every input shard for one output row must use the scale derived from the maximum absolute value across the complete unsharded row. TP1 and TP2 must therefore produce identical quantized weight shards and scales.

This affects more than standard `RowParallelLinear` instances. BAGEL stores secondary expert weights on a plain `Module`, so an `isinstance` check misses them. MiniMax-H3 uses a text encoder TP group that is independent from the DiT TP group, so reducing through the default vLLM TP group is also incorrect for that path.

Third, diffusion layers can pass rank-3 hidden states to the linear method. The ROCm Triton and AITER INT8 kernels accept a rank-2 matrix, so the common linear wrapper must flatten the leading dimensions before calling the upstream kernel and restore them afterward.

The proposed design is to keep the vLLM kernel selection and matrix multiplication unchanged. Upstream vLLM PR [#49764, Share online weight scales across TP](https://github.com/vllm-project/vllm/pull/49764), merged as commit `8170c23c4f`, fixes the same invariant for online FP8 and INT8 MoE. It detects whether the weight is sharded along a reduced dimension, computes the local maximum, and uses a `MAX` all-reduce before quantization.

vLLM-Omni still needs a small local adaptation because its diffusion linear path is not the upstream MoE path. The supplied static INT8 kernel accepts one scale, not one scale per output row, and MiniMax-H3 can use an independent text encoder TP group. The local code should therefore provide only the missing per-row packing and optional process-group override. Standard layers should continue to use vLLM's default TP group and all kernel selection and matrix multiplication should remain upstream-owned.

Validation on two ROCm GPUs with vLLM 0.27.0 covered the Triton kernel, the AITER kernel, exact TP1 versus TP2 quantized weight parity, MiniMax-H3 INT8 TP2 generation, and BAGEL INT8 TP2 generation. These were execution and kernel-correctness checks. They were not BF16 versus INT8 output-quality comparisons.

`VLLM_OMNI_LOGGING_LEVEL=DEBUG` was enabled for the recorded runs. There is no crash traceback because the scale defect produces incorrect numerical behavior rather than an exception.

The original diffusion INT8 implementation was added by @yjb767868009 in #1470. The unified quantization framework was added by @lishunyang12 in #1764. I am tagging them for context.
