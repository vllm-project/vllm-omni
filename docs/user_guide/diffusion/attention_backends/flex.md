# Qwen-Image 2.1 NPU attention paths

Qwen-Image 2.1 keeps `FLASH_ATTN` and its piecewise-span implementation as the
default on NPU. Compiled FlexAttention is an explicit prefill option within
that backend.

| `attn_path` | Prefill implementation |
| --- | --- |
| `auto` (default) | Piecewise spans when the backend supports them and there is no padding; otherwise dense mask |
| `piecewise` | FA Score with piecewise spans |
| `mask` | Explicit dense block-causal mask with FA Score (baseline) |
| `flex` | Compiled NPU FlexAttention with BlockMask size 128 |

All three block-causal implementations use the visibility rule
`(q >= kv) OR same_image_block`. Prefix-KV decode continues to use the existing
attention path; the Flex option applies to prefill only.

## Selection

For offline inference, use the existing sampling parameters:

```python
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

sampling_params = OmniDiffusionSamplingParams(extra_args={"attn_path": "flex"})
```

For image serving requests, set `"attn_path": "flex"` in the extra request
body. Use `"mask"` for the explicit-mask baseline, `"piecewise"` to force FA
spans, or omit the option to retain automatic selection. All requests in a
batch must choose the same path. Heterogeneous image batches require `auto`
or `mask`.

Direct transformer calls can use `attention_kwargs={"attn_path": "flex"}`.
Keep the attention backend set to `FLASH_ATTN`; there is no separate
`FLEX_ATTN` backend.

## Flex contract

Flex requires single-sample, unpadded BF16 NPU prefill, matching Q/K/V shapes,
no sequence parallelism and no attention quantization. Unsupported explicit
Flex requests raise an error. Compiler and execution errors propagate.

Use matching PyTorch, torch_npu, CANN and Ascend Triton builds with NPU
FlexAttention Inductor support, plus MindIE-SD for the FA paths.
`torch_npu._inductor` registers its Flex implementation before mask creation
and compilation. Only the Flex callable is compiled with
`torch.compile(..., backend="inductor", fullgraph=True, dynamic=True)`;
model eager mode does not disable this compilation.

BlockMask is built once per prefill and shared across transformer layers.
It remains in request metadata, with no global cache of request tensors.
Image boundaries are tensor data so changing lengths and image counts can
reuse the compiled callable. Measure first-use compilation separately from
warmed inference.

## Validation

CPU tests cover visibility, partial blocks, mask reuse, dynamic compilation,
request selection and attention dispatch. With repository test dependencies:

```bash
pytest -q tests/diffusion/attention/test_npu_flex_attention.py tests/diffusion/attention/test_flash_attn.py tests/diffusion/models/qwen_image_21/test_transformer_numerics.py tests/diffusion/models/qwen_image_21/test_attention_path.py -m 'core_model and cpu' --run-level=core_model
```

The weight-free NPU test compares real compiled Flex against an FP32 dense
reference across changing sequence lengths and image spans:

```bash
pytest -sv tests/diffusion/attention/test_flex_attn.py -m 'core_model and npu and A5' --run-level=core_model
```

CPU results do not establish NPU code generation, full-model correctness or
performance. Compare identical Q/K/V, then matching prompt/seed T2I and
multi-reference I2I against the explicit-mask baseline. Include at least two
denoising steps to exercise prefix-KV reuse. Measure mask creation,
compilation, warmed attention, generation latency and peak device memory
separately; skipped blocks alone do not establish a speedup over piecewise FA.
