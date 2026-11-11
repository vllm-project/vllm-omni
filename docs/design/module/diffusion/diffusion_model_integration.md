---
title: Diffusion Model Integration
kind: module
status: draft
owners:
  - "@fhfuih"
  - "@Bounty-hunter"
  - "@wtomin"
  - "@RuixiangMa"
  - "@timzsu"
primary_code_paths:
  - vllm_omni/diffusion/models/**
  - vllm_omni/diffusion/model_loader/**
related_code_paths:
  - vllm_omni/diffusion/layers/**
  - vllm_omni/diffusion/lora/**
  - vllm_omni/diffusion/utils/**
depends_on:
  - diffusion_runtime.md
  - ../input_output_modality_contracts.md
validation_paths:
  - tests/diffusion/models/**
  - tests/diffusion/model_loader/**
  - tests/diffusion/layers/**
  - tests/diffusion/lora/**
upstream_refs:
  - diffusers.DiffusionPipeline
last_reviewed: 2026-07-16
---

# Diffusion model integration

Diffusion model integration owns pipeline contracts, registration, checkpoint
loading, adapters, shared layers, and model-specific processing.

## Candidate invariants

### DIFF-MODEL-INV-001: Pipelines implement one runtime contract

**Rule:** A pipeline MUST declare its supported modalities, configuration,
inputs, outputs, loading path, and runtime capabilities.

### DIFF-MODEL-INV-002: Registration is the selection boundary

**Rule:** Runtime code MUST select model implementations through the registry or
loader contract rather than scattered model-name conditionals.

### DIFF-MODEL-INV-003: Model code does not schedule requests

**Rule:** Pipeline code MUST NOT own admission, batching, cancellation, or
cross-stage routing.

### DIFF-MODEL-INV-004: Shared behavior stays shared

**Rule:** Model directories SHOULD contain only genuine model differences.

## Safe-change guide

Test registry selection, checkpoint loading, minimal inference, input and output
contracts, and every declared optional capability.

### Hardware-specific block qualification

When a model selects native and vendor implementations of the same block,
qualify the extension seam before changing selection policy:

1. Load the same nontrivial state dict strictly into both implementations.
2. Exercise identity and channel-changing skip paths plus production mode
   branches.
3. Compare shape, dtype, and values with declared per-dtype tolerances.
4. Assert that the normal platform entry selects the intended class and that
   accelerated operations actually execute.
5. Restore process-global backend state after successful and failing calls.
6. Report block timings separately from model and request timings.
7. Record both the benchmark checkout revision and the imported package source
   path; an editable installation can otherwise execute a different checkout.
8. List the tested hardware and execution modes explicitly. Unrun
   `torch.compile` and CUDA Graph cells remain pending, not implicitly covered
   by eager-mode results.

Block qualification does not replace checkpoint loading, generated-output
quality, or request-level validation through the normal model entry.

For HunyuanImage3 ResBlocks, the initial evidence at `fa3d24d6` covers eager
execution on A100 80GB PCIe; an independent H200 eager reproduction was reported
by the reviewer on PR #7421. Neither establishes `torch.compile` or CUDA Graph
qualification. The production-sized BF16 fixtures failed the unchanged
`rtol=atol=0.02` gate at `e055a2d8` (33 passed, 2 failed in the full matrix).
The original smaller-fixture results do not qualify those geometries.

The production up/down cases cover the corresponding control-flow branches.
Both implementations use identity samplers there, so these cases do not
establish spatial upsampling or downsampling equivalence.

The qualification module's single-L4 resource marks match the CUDA ready lane's
model-test selection. They describe the CI target, not successful L4
qualification. L4 and current-source production-sized BF16 qualification remain
pending; A100 or H200 evidence must not be extrapolated to L4. The unchanged
numerical gate still applies to both production cases.

For numerical attribution without changing the benchmark's correctness gate:

```bash
python -m benchmarks.kernels.hunyuan_image3_resblock_benchmarks \
  --diagnose --dtype bfloat16 --case production_patch_embed \
  --output-json patch-embed-diagnostic.json
python -m benchmarks.kernels.hunyuan_image3_resblock_benchmarks \
  --diagnose --dtype bfloat16 --case production_final_layer \
  --output-json final-layer-diagnostic.json
```

These CUDA-only, untimed diagnostics reuse the qualification's original
geometry, parameter/input seeds, nonzero shared weights and tolerances. A failed
uninstrumented final comparison writes its JSON before exiting with status 1;
neither it nor intermediate agreement authorizes performance acceptance.
Ordinary benchmark mode still asserts correctness before timing.

The JSON separates propagated error in actual forwards from same-native-input
replays at GN/SiLU, conv1, embedding, adaptive GN/SiLU, conv2, skip and final
addition. It records native GN/modulation/scale/shift summaries and captured
cuDNN policies. GN and modulation inside fused kernels are not separately
materialized: the FP32 arithmetic probe is mechanism evidence, not a capture of
those internal values or an automatic explanation of the historical failures.
Convolution replays compare original policies, matched NVIDIA policy, and a
native-only policy change. TF32 settings are recorded but not changed for BF16.
The fusion-boundary replays also compare the actual native and fused outputs
against FP64 arithmetic on the same already-quantized inputs, weights and
conditioning values. `whole_block_fp64_reference` runs a private FP64 copy of
the native block on the input device, reporting native/default, native with
matched NVIDIA policy, and NVIDIA errors separately. This adds FP64 parameter
storage and slow untimed convolutions; budget resources accordingly.
FP64 absolute/RMS errors and tolerance counts describe reference error, not
the original-dtype compatibility gate or checkpoint-level quality. Being
closer to FP64 does not waive a failing native/NVIDIA comparison.
CPU copies and hooks make these runs unsuitable for timing; bitwise comparison
against uninstrumented forwards exposes capture/repeatability differences.
Use a dedicated process because cuDNN policies are process-global, although
temporary flags and hooks are restored on errors. Investigate mismatches and
rerun the unchanged GPU qualification matrix on the recorded source revision.

## Weight reloading

Pipelines that support live RL weight updates opt in with
`supports_weight_reload`. The shared loader integration records upstream vLLM
reload metadata before external checkpoint loading and postprocessing. Their
`load_weights` implementation must dispatch through parameter weight loaders so
upstream WTE can track layer completion and preserve live storage. See the
[weight-transfer contract](../../feature/weight_transfer.md#diffusion-loading-contract)
for support limits and validation requirements.

## Exact conditioning projection reuse

Models with repeated conditioning projections can use `ExactProjectionCache`
from `vllm_omni.diffusion.cache`. It stores actual projection outputs for identical
conditioning tensors and unchanged weights, with a byte budget and TP-wide hit
coordination. It has no model names, sampling schedules, projection layouts, or
artifact formats built in. MiniMax-H3 is its first model integration.

Keep one cache per model instance and prepare each conditioning tensor once
before its projections. For a leaf linear with fixed SiLU preprocessing:

```python
from torch.nn.functional import silu

from vllm_omni.diffusion.cache import ExactProjectionCache

cache = ExactProjectionCache(max_bytes=256 * 1024**2)

# Inside an inference forward, with gradients disabled:
cache.prepare(conditioning)
modulation = cache.project(
    "block.0.modulation",
    projection,
    conditioning,
    lambda: projection(silu(conditioning)),
)
```

The original computation must depend only on the supplied conditioning tensor,
the leaf projection's weights/buffers, and fixed preprocessing. Include prompt,
guidance, modality, or other request-dependent conditioning in the actual input;
a timestep-only key is unsafe when those affect the result. Do not mutate the
input between `prepare` and `project`, or modify outputs returned on cache hits.
Use separate cache instances for concurrent forwards with different inputs.

Call `clear()` when loading or switching adapters, changing projection behavior,
or moving the model. Weight-version checks invalidate individual entries.
Compilation, gradient-enabled execution, projection hooks, and unversioned
inference weights use the original computation. Validate cold/warm parity,
invalidation, and rank-local misses when integrating another model.
Hook checks include global hooks and hooks on wrapped submodules. Numerical
settings are captured at each projection call, so entering an autocast context
after `prepare()` cannot reuse an output from a different precision.

Validated offline results can be supplied through `_lookup_precomputed()`;
this extension is consulted only at TP1 on runtime-cache misses. Artifact
validation, weight binding, and model-specific layouts remain in the adapter.
H3's adapter retains its main/Ref2VA sidecars and their eager-only admission.

The cache retains projection results and leaves all original weights available.
It does not reduce weight residency or cancel layerwise prefetch. A future
integration that skips weight transfers must coordinate with `OffloadPlan`
before prefetch and provide a safe miss path; moving weight storage currently
causes conservative cache invalidation.
