# Platform Optimization Inheritance Reference

How a new diffusion model inherits vLLM-Omni's platform optimizations.
Worked example throughout: Boogu-Image (`#4995` -> `#6665` roadmap,
2026-06..09), an image-generation DiT (T2I Base + image-edit TI2I).

New diffusion models in vLLM-Omni inherit most platform optimizations
through **flags, shared ops, and three small interfaces** — not through
per-model reimplementation. This reference enumerates what is inherited
automatically, what requires a small documented contract, and what stays
model-owned, so model-side effort goes only where it pays.

Scope: the checklist is modality-general (video and audio DiTs inherit the
same way). Items marked **image-specific** below only apply when the final
output is an image.

## The two layers

**Platform layer (shared, do not reimplement):** attention backend selection,
response encoding, regional torch.compile, CFG-parallel framework, Ulysses
sequence-parallel framework, request batching, the component quantization
router, and the fused `qk_norm_rope` op. All of these live under
`vllm_omni/diffusion/` and are exercised by every model through flags or a
small integration surface.

**Model layer (owned, by design):** execution-semantics variants (distilled
student loops, DMD), model-unique fused kernels whose geometry is bound to
one checkpoint family, instruction-template design, and the model's own
evidence.

## Step-by-step inheritance checklist

### 1. Register the model (required)

Three registry sites plus metadata and docs — see the #6961 pattern:

- `_DIFFUSION_MODELS` in `vllm_omni/diffusion/registry.py`;
- `_DIFFUSION_PRE_PROCESS_FUNCS` / `_DIFFUSION_POST_PROCESS_FUNCS`;
- `_DIFFUSION_MODEL_METADATA_ALIASES` in `model_metadata.py` when the model
  shares a contract with a registered class (the `WanDMDPipeline` ->
  `WanPipeline` alias pattern), so admission limits like
  `max_multimodal_image_inputs` inherit correctly;
- `docs/models/supported_models.md` row + `recipes/` entry + index row.

### 2. Free wins on first serve (zero model code)

- **Attention backend:** the selector resolves a platform default, but the
  default is not always fastest. Validate one alternative
  (`--diffusion-attention-backend CUDNN_ATTN` measured -11.8% engine time vs
  the FLASH_ATTN default on SM90 for Boogu-Image; TRTLLM_ATTN requires
  SM100/103) and document the choice in the recipe.
- **Response format (image-specific):** `output_format`
  (`png`/`jpeg`/`webp`) and `output_compression` are supported per request;
  `jpeg` removes most of the response-encoding overhead (~134 ms at
  1024x1024). Document it for latency-sensitive users (#7446/#7447).
- **Regional torch.compile:** define `_repeated_blocks` on the transformer
  and the runner compiles the repeated regions automatically (per-region
  lazy compile; `dynamic=True` by default).

### 3. Shared ops and the three small interfaces (as appetite allows)

The fused op is a direct call; CFG parallelism, sequence parallelism, and
request batching are the three opt-in contracts the frameworks drive.

**a. Fused Q/K norm+RoPE (a few lines).** If the model's RoPE pairing is
adjacent-pair (interleaved) or half-split, call the shared
`fused_qk_norm_rope` op (`vllm_omni/diffusion/layers/fused_qk_norm_rope.py`)
with the `interleaved=` layout flag and the
`fused_qk_norm_rope_min_tokens` token gate, replacing the
`norm_q -> norm_k -> rotary -> cast` chain in every attention class. See
Boogu's `_qk_norm_rope` helper (#6982). Geometry bounds: any even
`rotary_dim <= head_dim <= 256`, bf16, CUDA+Triton; outside that, keep the
eager chain — gate on `_fused_cuda_supported` (bit-exact) rather than the
op's internal fallback (one-ulp different).

**b. CFG parallelism (~40 lines).** Mix in `CFGParallelMixin`
(`vllm_omni/diffusion/distributed/cfg_parallel.py`) and implement
`predict_noise(**kwargs)` plus `combine_cfg_noise(...)` preserving the
model's exact floating-point operation order (see the Boogu note: algebraic
rewrites introduce per-step drift). N-branch CFG adds
`combine_multi_branch_cfg_noise`. Then CFG=2 delivers ~1.6-2.0x on two GPUs
(Boogu: 1.9662x Base / 1.6029x Edit, pixel-exact).

**c. Sequence parallelism.** Provide a `_sp_plan`
(`vllm_omni/diffusion/distributed/sp_plan.py`) describing the token streams
and split boundaries; the framework handles head-count padding
(GQA-preserving: KV padded to a world-size multiple, Q derived by ratio),
the equal-pad all-gather skip, and the shard hooks. Boogu's plan covers
three streams (latents, reference, instruction) at one boundary.

**d. Request batching.** Set `supports_request_batch = True` on the pipeline
class (it is opt-in; the engine reads it via `getattr`) and accept a batch
in forward. T2I-only models should keep conditioned paths (Edit/Turbo) on
the single-request path until validated — false claims fail
the #6786-style pixel-parity checks.

### 4. Model-owned work (expected, not friction)

- Distilled / DMD / step-skipping execution variants: model-owned loop,
  mirror the upstream equations line-for-line and validate against the
  upstream reference (MAE/PSNR), as #6699 did.
- Model-unique fused kernels (modulation, exotic norms): follow the
  evidence-first pattern (#6607): per-op reference-vs-candidate timings, a
  numerical-equality result per op, and one complete-pipeline check; keep
  the kernels beside the model until #6305 defines a shared tier.
- Instruction-template / encoder-token design (biggest for image editing):
  DiT FLOPs scale linearly with the joint sequence length (for Boogu Turbo,
  ~3,000 of ~4,000 tokens are the fixed system template) — this is a
  model-design lever worth up to ~2.7x, independent of hardware and
  quantization.

### 5. Quantization (when ready)

Use the component router (`ComponentQuantizationConfig` in
`vllm_omni/quantization/component_config.py`, longest-prefix, fail-closed
manifest) — not a second routing layer; construct heavy components in their
target representation (no BF16-then-replace); keep `vae` explicit. See
the #6789 contract and #6791 for the construction-only first step.

## Evidence contract (mandatory, same for every model)

Follow the review-pr skill's
[perf-verification check](../../review-pr/references/checks/perf-verification.md):
one comparison column per claim at equal workload; runtime-switchable
changes A/B'd on the same head; P50/P100 from stated warmup/measured counts
with one population per number; named SHAs; an isolation statement;
tail-event accounting for cached or captured paths; and a quality gate (the
LPIPS <= 0.15 paired-comparison used by #6982 is the template for
fused-kernel defaults). New models also owe the sustained-cadence contract
check for realtime modes instead of perf claims.

## Sequencing advice

Flags and shared ops first (steps 2-3a) — they are small and review
quickly. CFG and SP next, each with its pixel/parity evidence. Quantization
last: it multiplies whatever kernel efficiency remains, and it is serialized
on the model's construction-time loading work.
