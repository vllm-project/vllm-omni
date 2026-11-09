# Qwen-Image 2.1 NPU attention paths

Qwen-Image 2.1 keeps `FLASH_ATTN` and its piecewise-span implementation as the
default on NPU. Compiled FlexAttention is an explicit prefill option within
that backend.

| `attn_path` | Prefill implementation |
| --- | --- |
| `auto` (code default) | Piecewise spans when the backend supports them and there is no padding; otherwise dense mask |
| `piecewise` | FA Score with piecewise spans |
| `mask` | Explicit dense block-causal mask with FA Score (baseline) |
| `flex` | Compiled NPU FlexAttention with BlockMask size 128 |

All three block-causal implementations use the visibility rule
`(q >= kv) OR same_image_block`. Prefix-KV decode continues to use the existing
attention path without Ring. When explicit Flex is combined with Ring, later
denoising steps also use Flex with full target-row visibility and cached prefix
K/V; this avoids bypassing the selected provider in the generic Ring runner.

## Selection

The pipeline owns attention selection through `_DEFAULT_ATTN_PATH` in
`pipeline_qwen_image_21.py`. The constant is set to `"auto"` and shared by the
`forward` and `prepare_encode` execution protocols. Change it to `"flex"`,
`"piecewise"`, or `"mask"` in model source to select another implementation.
Request `extra_args` does not expose `attn_path`.

The configured path applies to every request. Heterogeneous image batches
require `auto` or `mask`. When Flex is configured, the existing startup dummy
request uses the same path and performs the first compilation before serving.

Direct transformer calls can use `attention_kwargs={"attn_path": "flex"}`.
Keep the attention backend set to `FLASH_ATTN`; there is no separate
`FLEX_ATTN` backend.

## Flex contract

The Qwen integration requires single-sample BF16 NPU execution without attention
quantization. Single-device, Ulysses, Ring and hybrid Ulysses/Ring use the same
MindIE-SD Flex provider. Strict Ulysses requires divisible Q/KV head counts.
Unsupported explicit requests raise an error. Missing MindIE-SD or an older
package without the Flex APIs produces an actionable ImportError; transitive
dependency, compiler, allocation and execution errors retain their causes.

Use matching PyTorch, torch_npu, CANN and Ascend Triton builds with NPU
FlexAttention Inductor support, and a MindIE-SD build exporting
`FlexAttentionMask`, `parallel_flex_attention` and `make_block_causal_mask_mod`.
The existing image pin does not imply that a published wheel contains these
new APIs: install the corresponding MindIE-SD source build for validation.
MindIE lazily registers `torch_npu._inductor` before mask creation and
compilation. Only the Flex callable is compiled with
`torch.compile(..., backend="inductor", fullgraph=True, dynamic=True)`;
model eager mode does not disable this compilation.

Omni retains model selection, logical spans and request lifetime. Its existing
parallel strategy performs Ulysses exchange and output restoration. It passes
only the remaining Ring group to MindIE, avoiding duplicate communication.
MindIE rotates K/V and global position maps, includes replicated prefix K/V
once, and merges partial outputs using FP32 natural-log LSE. No FA Score or
piecewise implementation is moved to MindIE by this change.

For explicit NPU Flex only, Qwen pads targets and their RoPE frequencies before
the existing SP split hook. Global token validity hides padding; the existing
gather hook crops the output to the original target length. Other devices and
attention paths keep their existing padding policy.

BlockMasks are built per request and Ring block, then shared across transformer
layers. They remain in request metadata, with no global cache of request tensors.
Image boundaries are tensor data so changing lengths and image counts can
reuse the compiled callable. Measure first-use compilation separately from
warmed inference. Startup warmup executes two denoising steps, covering both
prefill and the Ring prefix-KV path. Different shapes, heads or rules can still
trigger new compilation; validate actual serving shapes and every rank.

## Validation

CPU tests cover visibility, partial blocks, mask reuse, dynamic compilation,
request selection and attention dispatch. With repository test dependencies:

```bash
pytest -q tests/diffusion/attention/test_npu_flex_adapter.py tests/diffusion/attention/test_npu_flex_ring_dispatch.py tests/diffusion/attention/test_flash_attn.py tests/diffusion/models/qwen_image_21/test_transformer_numerics.py tests/diffusion/models/qwen_image_21/test_attention_path.py -m 'core_model and cpu' --run-level=core_model
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

Run the MindIE-SD core and distributed tests from its checkout as well:

```bash
python -m pytest tests/UT/layers/flash_attn/test_flex_attention.py tests/UT/layers/flash_attn/test_flex_attention_parallel.py --test-mode=CPU -q
python -m pytest tests/ops/layers/flash_attn/test_flex_attention.py --test-mode=NPU -q
torchrun --standalone --nproc-per-node=4 -m pytest tests/ops/layers/flash_attn/test_flex_attention_parallel.py --test-mode=NPU -q
```

The four-rank suite covers pure Ulysses, pure Ring and their 2-by-2 hybrid,
global head-dependent scores, joint K/V and collective shape rejection. These
tests are provided for a matching NPU environment; static checks do not establish
that the multi-card implementation has passed device validation.
