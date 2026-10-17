# Diffusion attention execution contract PoC

This prototype accompanies [RFC #7226](https://github.com/vllm-project/vllm-omni/issues/7226)
and extends the [attention selection design](attention_backend_selection.md).
It demonstrates path-specific capabilities with dense BF16 FA4, NPU/ROCm
routing examples, and tensor-state lifetime with a test-only attention module.

## SDPA follow-up migration

The SDPA follow-up resolves CUDA dense noncausal FP16/BF16 calls after backend construction.
It distinguishes equal Q/KV head counts (`sdpa_equal_heads`), a PyTorch
fused-GQA probe accepting the original K/V head count (`sdpa_native_gqa`), and
the existing K/V repeat-interleave fallback (`sdpa_expanded_kv`). The resolver
and forward use the same probe on identically normalized masks and Q/K/V
layouts. A successful probe identifies the SDPA-level route; it does not
guarantee which fused kernel PyTorch chooses internally.

Only no-mask calls and 2D boolean key-padding masks with published
`attention_mask_mode="padding"` are marked supported. Packed, paged-KV,
piecewise, quantized-KV, causal, parallel, HSDP, non-CUDA, and unpublished mask paths
remain `UNMIGRATED`, so existing execution is not rejected. Pre-construction
capabilities are also `UNMIGRATED`, because the runtime GQA decision needs
actual tensors. Non-divisible Q/KV head ratios report `UNSUPPORTED` with the
same actionable error as forward.

Verified inference paths declare `TRACEABLE` for equal Q/KV heads and
`CUSTOM_OP` for GQA. The equal-head path passes a concrete boolean to SDPA;
the compiled CUDA GQA path wraps the runtime `SDPAParams` probe and native/
expanded-KV dispatch in an opaque custom op. Both eager execution and the
custom op use the same SDPA helper. The op transposes the SDPA result before
making it contiguous, returning BSHD directly to the caller. Its fake output
declares the same shape and strides, with V setting the output head dimension.
Flattening the output heads is a view rather than a subsequent layout copy.
Eager calls and non-CUDA entrypoints do not use this custom-op boundary.
Autograd fullgraph execution is not verified: inputs requiring gradients
retain `EAGER_ONLY` and bypass the inference-only custom op. A gradient-bearing
mask also bypasses that op. CUDA autocast capability reporting remains
`UNMIGRATED`, because route inspection on the original tensors does not
describe the cast inputs. Compiled execution explicitly casts eligible tensors
before the opaque boundary to preserve SDPA output dtype under autocast.

The CUDA regression matrix was rerun after rebasing onto merged PR #7379
(`596487fe6`) on an RTX 4090 (SM89, driver 595.71.05), Python 3.12.3,
PyTorch 2.13.0+cu130, and vLLM 0.30.0. It compares eager SDPA
output with explicitly expanded-K/V SDPA for BF16/FP16, equal-head and GQA
inputs, batches 1/2, head dimensions 64/512, masked/unmasked calls, and
square/non-square Q/K lengths. The complete focused SDPA and capability suite
passed 99 tests without exclusions; the ready-CI-style CUDA command passed
all 57 GPU cases. These results validate contract routing and numerics on this
environment, not a performance improvement or a cross-version compile claim.
The new CUDA test file uses the repository's L4 resource marker for CI routing
(also SM89) and explicitly skips when CUDA is unavailable; an unfiltered run
with CUDA hidden skipped all 57 GPU cases. CPU-only CI checks conservative
pre-construction and device mismatch but does not claim CUDA path coverage.

The fullgraph follow-up reproduced 12 failures before the change: equal-head
calls passed a `SymBool` to SDPA's boolean `enable_gqa` argument, and GQA calls
attempted to trace the pybind `SDPAParams` constructor. After the change, all
12 Inductor cases passed with `fullgraph=True, dynamic=True`, FP16/BF16,
masked/unmasked inputs, and two Q/K lengths with fresh data per case. Six
additional cases compile the production `Attention.forward` entrypoint and
verify its contract. Four `opcheck` cases validate schema, fake output, and
dynamic AOT dispatch. Twelve mixed-autocast cases check FP16/BF16 conversion
and additive-mask values; a mask-only-gradient case preserves the existing
eager fallback with graph breaks. With these and conservative capability
guards, the focused suite passes 137 tests on the same RTX 4090 environment. This
does not claim training, end-to-end model, CUDA-graph capture, or performance
coverage, nor guarantee one graph across arbitrary shapes.
The contiguous BSHD boundary may still require an output copy, depending on
the SDPA kernel's layout; realistic-shape performance impact has not been benchmarked.

The BSHD layout follow-up was validated on an A800 80GB PCIe (SM80, driver
595.71.05), Python 3.12.3, PyTorch 2.13.0+cu130, and vLLM 0.30.0. Four new
CPU cases check masked/unmasked outputs with equal and unequal Q/V head
dimensions, contiguous BSHD layout, numerics, and storage sharing when
flattening heads. Eight CUDA `opcheck` cases cover schema, fake strides, and
dynamic AOT dispatch, including unequal V dimensions. The existing dynamic
fullgraph cases also check contiguous GQA outputs and view-only head flattening.
The focused suite passed 145 tests; the ready-CI-style selection passed 99
CUDA cases on A800, and hiding CUDA skipped all 99 GPU cases. The L4 resource
marker remains CI routing and does not imply L4 hardware validation.

## Execution contract

`ExecutionContext` describes the requested execution path. `ExecutionPathResult`
reports its identity, support status and reason, and compilation mode.
`AttentionBackend.resolve_capabilities()` provides conservative pre-construction
results. After initialization, `Attention.resolve_execution_path()` supplies the
active parallel, paged-KV, and HSDP context; the backend combines it with the
selected kernel and normalized metadata.

Resolution runs outside compiled execution. FA4 reports `SUPPORTED` and
`CUSTOM_OP` for dense, noncausal BF16 without parallel or HSDP boundaries when
its kernel accepts the head dimensions. Dimension validation delegates to FA4's
architecture-specific rules through `backends/utils/fa.py`. Kernel rejections
report `UNSUPPORTED` with an actionable reason. Missing private validators and
other unmigrated paths report `UNMIGRATED` with advisory `EAGER_ONLY` defaults.
`requested_support()` checks a fullgraph request; it does not enforce selection
or change existing execution.

FA4 execution uses an opaque custom op. Its fake output preserves Q's batch,
sequence, and head count and uses V's head dimension. Output is contiguous even
when Q is noncontiguous. Metadata normalization is
shared with dispatch. Producers must update published mask semantics when masks
change; unpublished masks remain runtime-dependent to avoid synchronization.
Callers resolve again when execution metadata changes.

## NPU and ROCm worked examples

The same `FLASH_ATTN` backend resolves MindIE paths on NPU and AITER paths on
ROCm. Both examples report `UNMIGRATED`: their route is inspected, while hardware
correctness and compilation remain unvalidated. `EAGER_ONLY` is an advisory
default, not a measured compiler limitation. Existing execution is unchanged.

| Input | NPU / MindIE | ROCm / AITER |
| --- | --- | --- |
| Dense | `npu_dense` | `rocm_dense` |
| Padding mask | `npu_masked`: expand the key mask to `[B, 1, Q, K]` | `rocm_masked_varlen`: unpad, call varlen attention, then restore rows |
| Packed inputs | Opt-in `[real, pad]` contract; choose varlen or prefix-KV slicing from the environment | Forward complete packed-document boundaries to varlen attention |
| Incomplete packing | Rebuild a mask when possible; otherwise reject before the kernel | Shared CUDA-like normalization rejects incomplete metadata |
| Unpublished mask semantics | Route to the masked wrapper | Keep resolution runtime-dependent without reading mask values |

NPU resolution reuses `_resolve_packed_seq_npu`; ROCm reuses the metadata
normalization used by dispatch. AITER identity comes from initialization rather
than caller-provided context. Quantized, paged, piecewise, and parallel paths
remain outside these worked examples.

The new L1 tests use CPU tensors and replace only vendor kernel calls. They
check path identity, forwarded masks/boundaries, fallback rejection, and
conservative fullgraph-request handling. They do not test vendor numerics or
compiler support. Existing L1 CI already collects this test file.

```bash
python -m pytest tests/diffusion/attention/test_flash_attn.py \
  -k 'npu_contract or rocm_contract' -m 'core_model and cpu' -q
```

### Hardware validation required before migration

On an Ascend/MindIE or ROCm/AITER runner, record device, driver, PyTorch, vendor
library, and compiler versions, then validate the corresponding rows above:

1. Compare eager outputs with FP32 SDPA for dense and padding-mask inputs,
   including unequal Q/K lengths. Compare valid query rows; do not assume both
   platforms define padded-query outputs identically.
2. Compare packed outputs with separate per-document references. On NPU, cover
   both varlen and laser prefix-slicing modes, plus mask reconstruction and the
   missing-mask error. A padding fallback does not establish multi-document
   isolation. On ROCm, cover multiple real documents.
3. Check output shape, dtype, device, finite values, and unchanged inputs.
   Start BF16 comparisons at `atol=rtol=1e-2` and have the platform maintainer
   confirm the tolerance against its reference tests.
4. Run `torch.compile(fullgraph=True, dynamic=True)` with the platform's supported
   compiler over multiple sequence lengths. Record graph breaks and recompiles;
   choose `TRACEABLE`, `CUSTOM_OP`, or verified `EAGER_ONLY` from that evidence.
   A custom-op path additionally needs schema/fake checks and repeated replay.

Promote only the paths validated on that platform. These hardware checks have
not been run locally; ROCm/NPU fullgraph support is not claimed.

## State lifetime example

`test_attention_state_lifetime.py` prepares an owned query-scale tensor eagerly
and passes it explicitly to an opaque attention op. The compiled module retains
its state after the caller drops its reference and releases ownership when the
compiled callable is deleted. Equivalent instances reuse a graph with their own
state values.

This example tests ownership and release, dynamic replay across instances,
custom-op schema/fake behavior, and Inductor execution. It adds no production
backend or state registry. Planning keys, caching, fallback-policy declarations,
opaque handles, CUDA graph capture, and TRTLLM integration are deferred.

## Validation

```bash
python -m pytest tests/diffusion/attention/test_flash_attn_compile.py \
  tests/diffusion/attention/test_flash_attn.py \
  tests/diffusion/attention/test_attention_capabilities.py \
  tests/diffusion/attention/test_attention_state_lifetime.py -q -rs
```

The focused suite covers capability decisions and metadata changes, existing
padding/mask regressions, tensor-state lifetime, and real FA4 fullgraph execution
against FP32 SDPA and eager FA4. Representative Q/K and V dimensions are
(32, 32), (64, 64), (80, 48), (192, 128), and (256, 256), with batches 1 and 2
and sequence lengths up to 1,024. Invalid dimensions are checked against actual
kernel errors. These samples exercise the contract rather than define support.
BF16 comparisons use `atol=rtol=1e-2`; `torch.library.opcheck` checks schema and
fake-output correctness.

Single-graph reuse is asserted in the tensor-state test. The FA4 numerical tests
retain compiler caches across shapes within each case but do not require one
graph across different batch sizes, Q/K length equality, or head dimensions.

Tested environment: GB300, PyTorch `2.13.0+cu130`, FA4 `4.0.0b18`, CUTLASS DSL
`4.6.2` with CUDA 13 libraries, Quack `0.6.4`, and TVM FFI `0.1.11`.
Real-kernel tests require CUDA and CuTe FA4; their availability is checked at runtime.

Before the rebase, the focused suite passed all 97 tests without skips, including
11 CPU NPU/ROCm routing tests. After rebasing onto upstream `e3be42e05`, the
expanded suite reports 99 passed and 11 failed in the local vLLM 0.28 environment.
All 11 failures are import errors: upstream now requires `compute_layout_strides`
from vLLM, which this environment lacks. Both noncontiguous FA4 regression cases
and real-kernel schema/fake checks pass. The full suite must be rerun with the
vLLM version required by upstream. NPU/ROCm device numerics and compilation remain
unvalidated. Pre-commit passes with the CI hook skips. Upstream PyTorch/CUTLASS
warnings remain.

## SageAttention dense execution

`SAGE_ATTN` declares `SUPPORTED` / `CUSTOM_OP` for dense FP16/BF16 on
H100/H200/GH200 (SM90), head sizes 32/64/96/128 with contiguous head elements, and equal Q/K/V head
counts. Causal attention additionally requires equal Q/K lengths. Resolution
uses the input device and initialized causal setting. Other architectures, XPU,
packed, paged-KV, piecewise, quantized-KV, parallel, and HSDP paths remain
`UNMIGRATED`; the pre-construction result is also conservative.

`test_sage_attn_compile.py` validates real eager/fullgraph replay, padded-head
output strides, input preservation, schema/fake agreement, and the production
`Attention` entry point.

### SageAttention3

SageAttention3 uses a separate backend and custom op from SageAttention2.
Its verified `SUPPORTED` / `CUSTOM_OP` scope is dense FP16/BF16 on SM120,
head sizes 64/128 with contiguous head elements, equal Q/K/V head counts,
default softmax scale, and no dropout. Causal attention requires equal Q/K lengths.
Packed, paged-KV, piecewise, parallel, HSDP, and other unverified configurations
remain `UNMIGRATED`, including SM121. This does not prevent ordinary execution
or compilation. Head size 256 is not claimed
as an FP4 path: the inspected Sage3 dispatcher falls back to SDPA there.
The custom op copies K before vendor preprocessing (which centers K in place)
and normalizes output strides to match its fake implementation.
`test_sage_attn3_compile.py` targets upstream runtime architectures SM120/SM121
with a matching Sage3 binary, checking input preservation,
full-graph replay against eager Sage3, and schema/fake agreement for head sizes
64 and 128.

## TRTLLM dense execution

`TRTLLM_ATTN` declares `SUPPORTED` / `CUSTOM_OP` for noncausal dense BF16 on
B200/GB200 (SM100) and B300/GB300 (SM103), head dimension 128, and equal Q/K/V
head counts, without parallel, paged-KV, piecewise, or HSDP boundaries. Pre-construction, SAGE, skip-softmax, packed,
and other unverified paths remain `UNMIGRATED`. Resolution and dispatch share
metadata validation; workspace mutation is explicit in the custom-op schema.

CPU contract tests cover fullgraph replay and schema/fake consistency using a
substituted dispatcher. Real-kernel validation requires Blackwell with FlashInfer:

```bash
python -m pytest tests/diffusion/attention/test_trtllm_attn.py \
  -k dense_contract_fullgraph_matches_sdpa -q -rs
```

These hardware tests compare eager/compiled output with FP32 SDPA and check
schema/fake agreement. The TRTLLM and contract suite passed on Blackwell before the final architecture
restriction was added: 61 passed, no skips (19 dependency deprecation warnings).
The SM100/SM103 restriction is additionally covered by CPU contract tests.
