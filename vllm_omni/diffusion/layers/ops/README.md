# Shared diffusion operations

Models and layers import shared tensor operations from
`vllm_omni.diffusion.layers.ops`. Parameters, input preparation, and model
enablement policies remain with the caller. This is the existing-operation
migration portion of [RFC #7382](https://github.com/vllm-project/vllm-omni/issues/7382),
not completion of its cross-model qualification work.

## Q/K RMSNorm + RoPE

```python
from vllm_omni.diffusion.layers.ops import fused_qk_norm_rope
```

The canonical module is `rope/qk_norm_rope.py`. It owns the kernels, eager
reference, backend predicates, dispatch, and the schema/fake implementation for
`torch.ops.vllm_omni.fused_qk_norm_rope`. The old
`vllm_omni.diffusion.layers.fused_qk_norm_rope` module re-exports the same public
function; it does not register or wrap another operation. Internal implementation
tests must patch the canonical module, not the compatibility module.

| Contract | Supported behavior |
| --- | --- |
| Inputs | Separate Q/K `[tokens, heads, head_dim]`; token count, head dimension, dtype and device match; head counts may differ |
| Weights | Separate `[head_dim]` norm weights on the activation device; owned by the model |
| RoPE table | `[tokens, rotary_dim]`, packed half-width `[cos \| sin]`, same device and dtype as Q/K |
| Semantics | Per-head RMSNorm followed by half-split RoPE; trailing non-rotary dimensions remain normalized |
| Valid geometry | Positive even `rotary_dim <= head_dim`; FP32, FP16 or BF16 inputs |
| Outputs | New Q/K tensors; input tensors are not mutated |
| CUDA acceleration | Triton, BF16, `head_dim=128`, `rotary_dim=96` |
| NPU path | Existing H3 BF16 128/96 path using Ascend RMSNorm and rotary primitives |
| Other valid inputs | Existing `F.rms_norm` and tensor RoPE reference |
| Invalid inputs | Existing public validation raises; runtime failures are not caught by a new fallback |

The current production consumer is MiniMax-H3's `MiniMaxH3Attention` in
`diffusion/models/minimax_h3/minimax_h3_transformer.py`. This migration changes
its import only. H3's weights, table preparation, and no-table normalization
path remain model-owned. The CUDA torch registration and direct NPU/eager
dispatch are preserved.

## Validation and follow-up

- `tests/diffusion/layers/ops/test_qk_norm_rope_imports.py`: fresh-process import
  order/identity, CPU reference behavior on packed GQA views, input validation,
  and registered fake output metadata.
- `tests/diffusion/layers/test_fused_qk_norm_rope.py`: existing CUDA BF16 reference
  comparison (`atol=0.0625`, `rtol=0.02`), fullgraph compilation with input
  preservation, and CUDA Graph replay after updating input buffers.
- `tests/diffusion/layers/test_fused_qk_norm_rope_npu.py`: CPU tests of the Ascend
  adapters and real-device dispatch/reference coverage (`atol=rtol=0.02`).

Hardware tests require their corresponding runtime. CPU adapter tests do not
qualify NPU execution. Record hardware, dependency versions, actual commands,
and results with each PR; source migration alone does not establish model-level
accuracy or latency non-regression. Claimed compile/graph modes and real H3
outputs/timings need separate verification under the RFC's Q1/Q2 criteria.

The CUDA test file is collected by the existing **Diffusion · Other Test**
job in `.buildkite/cuda/test-ready.yml`. Run the same operator checks locally
with a CUDA GPU and the matching vLLM/Omni environment; no model weights are
required:

```bash
python -m pytest -q tests/diffusion/layers/test_fused_qk_norm_rope.py \
  -m 'core_model and cuda' --run-level=core_model
```

The proposed integration order keeps this H3 migration first, followed by
Boogu's interleaved kernel, token gate, and model integration in
[PR #6982](https://github.com/vllm-project/vllm-omni/pull/6982). Complementary
public support-query work in
[PR #7422](https://github.com/vllm-project/vllm-omni/pull/7422) then follows the
adopted Boogu revision. Those additions are outside this migration. Preserve
Boogu's own enablement policy and reference when integrating that work; the
shared eager reference is not a universal replacement for every model's
normalization chain.
