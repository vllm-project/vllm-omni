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
The legacy module also preserves `fused_qk_norm_rope_min_tokens` and the private
`_fused_cuda_supported` alias used by the merged Boogu integration. Their
implementation remains canonical; this migration adds no public support query.

| Contract | Supported behavior |
| --- | --- |
| Inputs | Separate Q/K `[tokens, heads, head_dim]`; token count, head dimension, dtype and device match; head counts may differ |
| Weights | Separate `[head_dim]` norm weights on the activation device; owned by the model |
| RoPE table | `[tokens, rotary_dim]`, packed half-width `[cos \| sin]`, same device as Q/K; activation dtype or FP32 |
| Semantics | Per-head RMSNorm followed by half-split RoPE (default) or adjacent pairs (`interleaved=True`); trailing non-rotary dimensions remain normalized |
| Valid geometry | Positive even `rotary_dim <= head_dim`; FP32, FP16 or BF16 inputs |
| Outputs | New Q/K tensors; input tensors are not mutated |
| CUDA half-split acceleration | Existing Triton BF16 kernel, `head_dim=128`, `rotary_dim=96` |
| CUDA interleaved acceleration | Merged Boogu Triton BF16 kernel, even `2 <= rotary_dim <= head_dim <= 256` |
| NPU path | Existing H3 half-split BF16 128/96 path using Ascend RMSNorm and rotary primitives |
| Other valid inputs | Existing `F.rms_norm` and tensor RoPE reference |
| Invalid inputs | Existing public validation raises; runtime failures are not caught by a new fallback |

The public package consumer is MiniMax-H3's `MiniMaxH3Attention` in
`diffusion/models/minimax_h3/minimax_h3_transformer.py`. This migration changes
its import only. H3's weights, table preparation, and no-table normalization
path remain model-owned. The CUDA torch registration and direct NPU/eager
dispatch are preserved.
Boogu continues using the legacy imports and its existing token gate, private
CUDA predicate, FP32 rotary tables, and eager chain from merged PR #6982.
The interleaved kernel and eager reference retain FP32 rotation of the
activation-dtype rounded normalized values; H3 keeps its original arithmetic.

## QK/RoPE contract detail (RFC #7382 P0)

Adopted sources: [#5990](https://github.com/vllm-project/vllm-omni/pull/5990)
(merged 2026-08-14, H3 half-split 128/96 contract),
[#6982](https://github.com/vllm-project/vllm-omni/pull/6982) (merged as
`f7d9deb4`, interleaved variant and token gate), and this migration (#7417).
The five categories follow
[RFC #7382 §3](https://github.com/vllm-project/vllm-omni/issues/7382#user-content-integration-rules).

### 1. Tensor and numerical behavior

| Item | Contract |
| --- | --- |
| Q/K | Separate `[tokens, heads, head_dim]` tensors; tokens, `head_dim`, dtype and device must match; head counts may differ (GQA) |
| Norm weights | Separate `[head_dim]` tensors on the activation device; model-owned |
| Epsilon | Model-passed RMSNorm epsilon; the same value reaches every path (kernel, eager reference, NPU) |
| RoPE table | `[tokens, rotary_dim]` packed half-width `[cos \| sin]`; same device; activation dtype or FP32 |
| Valid dtypes | BF16, FP16, FP32; accelerated paths are BF16-only |
| Valid geometry | Even `rotary_dim`, `2 <= rotary_dim <= head_dim`; the non-rotary tail stays normalized |
| Layout | Q/K strides are passed to the kernel (non-contiguous supported); weights and the table are made contiguous by the public entry |
| Rounding, half-split kernel | Normalized values rounded to BF16, rotation in FP32 |
| Rounding, half-split eager reference | Historical x-dtype arithmetic |
| Rounding, interleaved (kernel and reference) | FP32 rotation of the BF16-rounded normalized value; fused vs eager within one operand ulp (≈28% of Q/K elements differ by exactly one BF16 ulp) |
| Tolerances | CUDA `atol=0.0625, rtol=0.02`; NPU `atol=rtol=0.02` (own suite, not CUDA's) |
| Mutation/aliasing | Outputs are new tensors; inputs are not mutated |

### 2. Execution behavior

- Supported accelerated variants: half-split exactly `head_dim=128, rotary_dim=96`
  (per-tensor kernel); interleaved even `2 <= rotary_dim <= head_dim <= 256`
  (combined Q+K kernel); NPU half-split 128/96 (Ascend primitives). Dispatch
  order: NPU → CUDA fused → eager reference. Half-split production traffic
  keeps the per-tensor kernel; the combined kernel's half-split mode is
  test-covered but not routed.
- Reference behavior: `_apply_rope_table` + `_eager_qk_norm_rope` serve as the
  fallback and the unit-test golden.
- Compile/graph modes: `torch.compile` fullgraph and CUDA-graph replay
  supported via the registered op and its fake implementation.
- Build/resource recovery: the NPU path has no silent eager fallback — a
  missing MindIE-SD falls back to `npu_rotary_mul`, not to eager; without
  CUDA + Triton, the eager reference serves.
- Execution-error handling: changes to error handling require separate
  justification and path/state tests.
- Relocation preserves behavior: the canonical module is AST-identical to the
  accepted implementation.

### 3. Reference and errors

- Valid inputs outside the supported accelerated variants use the
  op-owned eager reference.
- Invalid inputs raise (`ValueError` / `TypeError`) before dispatch
  (shape/geometry/device mismatches raise `ValueError`; non-floating dtype
  raises `TypeError`), and runtime failures are not caught by a new
  fallback — retaining a model reference does not imply catching every
  kernel failure.
- Two fallback layers, implemented separately: the op owns the shared eager
  reference; each consumer whose original chain rounds differently keeps its
  own model-level original chain (e.g., Boogu, bit-exact on gate rejection).

### 4. Public API

- Keyword-only `head_dim/rotary_dim/interleaved`; `interleaved` is a
  model-passed variant choice, not an operator decision.
- Support query only when a caller needs it: `can_use_fused_qk_norm_rope`
  lands with #7422; #7417 adds no public support query.
- Optional dependencies stay lazy (`import torch_npu` inside the function,
  `find_spec("mindiesd")` probe; no eager fallback when MindIE-SD is missing).
- Single torch registration owner in the canonical module; the legacy module
  re-exports only.
- Metadata recording stays separate from per-call dispatch: contract,
  consumer, and result records live in this README and linked documents,
  not in the dispatch code path.
- The token-gate helper stays out of the public package;
  `VLLM_OMNI_FUSED_QK_NORM_ROPE_MIN_TOKENS` overrides per deployment.

### 5. Adoption

| Consumer | Public import | Model-owned pieces |
| --- | --- | --- |
| MiniMax-H3 | `layers.ops` (import-only change in #7417) | norm weights, table preparation, no-table normalization path; `interleaved=False` |
| Boogu-Image | legacy imports; public-surface adoption in #7422 | packed `[cos\|sin]` table materialization; token gate policy (`B*S >= 2048` default, env-overridable, permanent through #7422's public-surface migration); model-owned fallback chain; `interleaved=True` |

## Qualification status (RFC #7382)

| Consumer | Configuration | Status / evidence |
| --- | --- | --- |
| MiniMax-H3 QK/RoPE | CUDA; BF16; unit fixture `tokens={1,257,1024}`, 14 heads, head 128 / rotary 96, eps `1e-5`, packed QKV views; eager + supported compile/graph modes; baseline = pre-move fused implementation | Op-level checks pass on A100 80GB: 43 tests, import parity across fresh processes ([report](https://github.com/vllm-project/vllm-omni/issues/7382#issuecomment-5661896913)). Model-level profile pending: proposed 1344×768, 50 steps, SSIM ≥ 0.97, PSNR ≥ 34 |
| H3 NPU QK/RoPE | Existing NPU fixture shapes/dtypes; device and runtime TBD | Pending NPU resources; not claimed as a validated migration here |
| Boogu self/joint QK/RoPE | BF16 interleaved; production geometry 120/120 (28 Q heads, 7 KV heads); carried #6982 workload: 1024×1024, 28/50 steps; `B*S=2048` boundary below/at/above | A100 pilot reproduces the boundary (2047 falls back, 2048/2049 fuse), bit-exact forced fallback, compile + graph pass ([report](https://github.com/vllm-project/vllm-omni/issues/7382#issuecomment-5661555514)). Final rerun after #7417/#7422 stabilize |

Remaining gaps: MiniMax-H3 model-level profile confirmation, NPU device, Boogu
final checkpoint/seed/tolerance pinning. The validation runs stay with their
current owners.

## Validation and follow-up

- `tests/diffusion/layers/ops/test_qk_norm_rope_imports.py`: fresh-process import
  order/identity, CPU reference behavior on packed GQA views, input validation,
  and registered fake output metadata.
- `tests/diffusion/layers/test_fused_qk_norm_rope.py`: existing CUDA BF16 reference
  comparison (`atol=0.0625`, `rtol=0.02`), fullgraph compilation with input
  preservation, and CUDA Graph replay after updating input buffers. The merged
  interleaved/general-geometry tests and token-gate tests are retained.
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

The integration order is merged
[PR #6982](https://github.com/vllm-project/vllm-omni/pull/6982), this canonical
migration, then complementary public support-query and Boogu public-surface
adoption in [PR #7422](https://github.com/vllm-project/vllm-omni/pull/7422).
This migration relocates the accepted kernels without changing their behavior.
Preserve Boogu's own enablement policy and reference in the follow-up; the
shared eager reference is not a universal replacement for every model's
normalization chain.
