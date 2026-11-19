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

Qwen-Image-Edit-Plus initializes its Qwen2.5-VL encoder on CPU before sharding
and moving it to the execution device. Its non-persistent FP32 RoPE frequency
buffers must match Diffusers' CPU initialization: computing them on the
accelerator can change BF16 conditioning and image accuracy. The DiT retains
the loader's device context and the selected compilation and attention modes.

HunyuanImage3 uses `OmniDiffusionConfig.revision` for model configuration,
generation configuration, tokenizer, and weight loading, including preprocessing.
The shared-weight runtime pins remote loads to the commit resolved by AR;
local checkpoint directories must remain immutable throughout the session.

## HunyuanImage3 AR KV token identity

The HunyuanImage3 AR-to-DiT bridge carries the original prompt and generated
token IDs alongside display text. Scheduler preparation and worker execution
use those generated IDs for CoT sections, then build the uncached image suffix
with the existing template. Decoding and re-encoding generated text is not an
identity operation, even when the token count stays the same.

Before importing AR KV, the conditional prefix must match the original AR
sequence token by token. Dense reuse is bounded by the actual key and value
coverage of every decoder layer; native transfer uses confirmed computed-token
coverage. The closing CoT token remains outside the reused prefix. Dropping
cached thought tokens or receiving incomplete image/CFG coverage is rejected
before injection. Full recomputation can still drop thoughts. Standalone text
requests and AR outputs without a completed CoT retain their template path.

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

## MiniMax-H3 approximate reference reuse

The request-local reference-KV cache is distinct from exact conditioning
projection reuse. `kv_storage=host` retains the existing layerwise prefetch;
`kv_storage=npu` keeps unquantized reference KV on the device. Both refresh on
the configured interval and compact reference query rows on reuse steps. The
device pool and concatenation scratch are read and overwritten on the compute
stream in layer order and released when the request closes.

Reference reuse does not install an adapter or change weight loading. A request
keeps its model and adapter fixed for the denoise run; its cache closes before
another request can reuse storage. Refresh intervals are independent of the
step count. Per-image reference selection is a separate RainFusion option and
requires exact image ownership metadata. See the
[reference optimization recipe](../../../../recipes/h3_ref_optimizations/README.md)
for configuration, its independent MindIE-SD dependency and validation limits.

### Fixed eight-step Ref2VA adapter

The separate LQ-Slim preset accepts the immutable v4-step600 EMA artifact through
the normal startup LoRA flags. It adds BF16 deltas after base projections,
including quantized projections. Its hash and 259-target layout are validated
before binding. Installation clears the AdaLN projection cache; forward hooks
retain that cache's normal conservative fallback. The adapter requires resident
DiT weights, scale 1, eight steps and DiT TP=1; it disables per-request adapter
switching. These restrictions do not apply to the independent Ref extensions.
See the [LQ-Slim recipe](../../../../recipes/lq_slim/README.md).
