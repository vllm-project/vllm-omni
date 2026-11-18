# Diffusion Attention Backend Selection

This document defines the selection and extension contract for diffusion
attention backends. User-facing backend choices, installation, and tuning are
in the [attention backend guides](../../user_guide/diffusion/attention_backends.md).

The [execution contract PoC](attention_execution_contract_poc.md) describes the
experimental post-selection capability and lifecycle contract from RFC #7226.

## Scope

The contract applies to DiT and other diffusion attention layers. It is
separate from vLLM's autoregressive attention selector.

The selection path has four responsibilities:

1. normalize user configuration into `AttentionConfig` and `AttentionSpec`;
2. resolve a spec for an attention role;
3. ask the active platform to validate an explicit backend or choose a
   hardware default; and
4. load the selected `AttentionBackend` class from the registry path.

## Resolution contract

`get_attn_backend_for_role()` resolves configuration in this order:

1. exact `per_role[role]`;
2. category `per_role[role_category]`;
3. `default`;
4. platform default.

An explicit resolution returns both the backend class and its
`AttentionSpec`. A platform-default resolution returns the class and `None`.
Layers must therefore treat the spec as optional and must not infer that a
missing spec means SDPA.

Class resolution is cached by backend name, head size, and whether TRTLLM may
be selected automatically. Logging is cached separately by role and source so
the startup record identifies why each role received its backend.

## Registry and platform boundary

`DiffusionAttentionBackendEnum` maps stable configuration names to default
qualified class paths. `register_diffusion_backend()` may replace a path at
runtime without changing the public enum value.

The active `OmniPlatform` owns compatibility policy through
`get_diffusion_attn_backend_cls()`. It must:

- validate explicit selections and fail with an actionable error when the
  requested kernel cannot run;
- choose only an available, compatible default when no backend is explicit;
- account for head size and other platform-visible constraints; and
- return a qualified class path implementing `AttentionBackend`.

Hub preflight and execution share the same version resolver. FlashAttention 2
tries versions 1, 2, then 3 so Torch stable-ABI builds remain available when
older per-Torch builds do not match. An explicit selection still fails when
no compatible published build resolves.

Explicit backend validation must avoid initializing unrelated runtimes. The
cuDNN version probe belongs to automatic Blackwell fallback selection.

The selector must not duplicate device capability or package-availability
policy that belongs to the platform.

## Typed backend options

Backend-specific settings remain typed fields on `AttentionSpec` rather than
unstructured keyword dictionaries:

- `quant` is consumed by FlashInfer and TRTLLM with backend-specific value
  validation;
- `skip_softmax` is consumed by TRTLLM; and
- `block_sparse` is consumed by block-sparse backends such as RainFusion; and
- `fastvideo_vsa_topk` is consumed by FastVideo VSA; and
- `sparge` is consumed by the XPU SPARGE INT8 sparse attention backend; and
- `sol_attn` is consumed by the CUDA Sol-Attn backend. It accepts a single
  noncausal BF16 sequence with equal Q/KV heads and head dimension 128.

Sol-Attn consumes producer-owned `PackedPaddingMetadata` to trim structural
suffix padding before both dense and sparse work. Arbitrary packed documents
are rejected rather than inferred from maximum sequence lengths. Its optional
`prefix_sink` in explicit `sink_mode="prefix"` uses model-owned video layout to protect the prefix as keys and
as dense queries. The sparse kernel has an opaque custom-op compilation boundary;
Ring and AllGather-KV cannot bypass its full-sequence contract.
An explicit sink that clamps to an empty range does not enlarge the protected
video prefix. For the explicit native provider, `kv_splits="auto"` resolves to four splits for SM90 with CuTe and
at least 65,536 valid tokens, and one otherwise, matching upstream's integration
policy. Explicit multi-split requests still require SM90 and the CuTe runtime.

A backend reads only the fields it owns and rejects incompatible values. New
options should be added to a shared typed spec only when more than one backend
shares their semantics; otherwise add a dedicated typed block.

## Model integration contract

Each diffusion attention call declares a stable role and, when appropriate, a
broader category. Model-specific roles permit precise overrides; categories
preserve common `self` or `cross` policies. A model also decides whether its
path is compatible with automatic TRTLLM selection, because only the model
knows whether masking and packing satisfy the kernel contract.

### Model-owned implementation specializations

Use `Attention(..., impl_overrides={backend_name: implementation_class})`
when a model needs different computation within an existing backend contract.
The model defines the implementation in its own directory and passes it when
constructing the layer:

```python
# MyModelVSAImpl is a model-owned subclass of FastVideoVSAImpl.
self.attention = Attention(
    num_heads=num_heads,
    head_size=head_size,
    softmax_scale=head_size**-0.5,
    causal=False,
    role="self",
    impl_overrides={"FASTVIDEO_VSA": MyModelVSAImpl},
)
```

The override is applied after role/platform selection, keyed by the selected
backend's `get_name()`. It must subclass the **selected implementation**;
incompatible platform implementations raise `TypeError`. Unselected overrides
have no effect. Backend capabilities, constructor options, and shared
parallel dispatch remain unchanged and must be honored by the specialization.

For explicit block-sparse attention, the model owns tiling, routing and any
compression correction. The shared provider layer consumes that mask, its
logical block size and valid-token counts. It owns kernel availability,
provider-specific sparse metadata, transport padding and execution. The cuDNN
provider consumes tile64 or tile128 directly; selecting it does not change
the model's top-k policy. Startup checks and execution use the same provider
resolution path. The cuDNN adapter imports the public
`cudnn.block_sparse_attention` namespace; it must not depend on the legacy
`cudnn.torch` re-export removed in frontend 1.32.

The shared `block_sparse_attn_bshd(..., return_lse=True)` contract exposes
the selected blocks' normalization state for algorithm-owned residual merges.
It returns BSHD output and contiguous FP32 **natural-log** LSE in `[B,H,Sq]`
layout. Empty rows have zero output and negative-infinite LSE. Only explicit
`cudnn` and `triton` BF16 providers currently implement this contract; other
provider/precision combinations fail rather than silently discarding LSE or
switching kernels. The Triton adapter converts its kernel's base-2 LSE;
cuDNN already returns natural-log LSE. The default remains output-only.

This normalization contract does not make SOL and VSA interchangeable.
SOL needs its own training-free routing and a normalization-weighted merge
of exact selected blocks with omitted-block centroid approximations. H3 VSA
retains its checkpoint-owned compressed-attention gate. Selecting `SOL_ATTN`
defaults to `provider="cudnn"`, `sink_mode="text_audio"`, `tau=1`, and
`thresh_type="diag"`, including when the `sol_attn` options are omitted.
This uses the upstream Sol-H3 Ref2VA protection policy for all supported H3
tasks, not its LoRA, quantized transport, or shortened sampling schedule.
The platform's default backend is unchanged. Explicit `provider="native"`
with `sink_mode="prefix"` retains the native fused path; `provider="triton"`
selects the other BSA provider. Both BSA providers use SOL, not VSA routing: upstream SOL
preprocessing supplies BF16 K means, BF16 V sums, and base-2 thresholds;
the adapter routes pooled Q against K means and protects neighboring tiles.
Sink tiles are dense along both query and key axes, so they do not require
a second dense-query pass. This rounds sink-query protection up to complete
64-token tiles, unlike the native path's token-range overwrite.

The default `sol_attn.sink_mode="text_audio"` replaces the prefix/range policy with a
producer-owned `AttentionMetadata.text_audio_block_mask`. Its `TextAudioBlockMask`
records valid length, block size, and a contiguous Boolean vector on the operand
device. Only block64 metadata matching the global used sequence is accepted,
including during dense guards. The producer validates nonempty text/audio
positions once per request; consumers do not read device scalars per layer.
Both Q rows and KV columns of marked blocks stay exact. Reference video is not
implicitly protected, apart from mixed boundary blocks. This mode requires a
BSA provider and rejects explicit sink ranges; it overrides `prefix_sink`.
Pure Ulysses preserves the
global mask; compact unequal-Q/KV and multi-document layouts remain unsupported.
The defaults `dense_steps=0` and `dense_layers=[]` permit BSA at every main-DiT step and layer,
but do not disable exact sink/neighbor blocks or change the sampler step count.

The SOL-BSA residual uses query chunks and the provider's natural-log LSE.
Omitted-block token counts weight the denominator, while V sums already
contain their numerator multiplicity. It is a correctness-first PyTorch
composition, not a fused residual kernel or a validated speedup. Routing
metadata still scales quadratically in the number of tiles. BSA providers
resolve `kv_splits="auto"` to one and reject explicit multi-split requests;
Triton additionally requires standard head-dimension scaling. Provider and
preprocessor dependencies are checked at construction and for the operand
device at execution. Errors do not fall back to native SOL or dense attention.

Qwen-Image-2.1 publishes a zero-based denoising index in
`ForwardContext.denoise_step_idx` before running the transformer. Both CFG
branches see the same index. In step execution, Qwen defers new first-step
requests while a batch is decoding. The runner advances each scheduled request
once per iteration, so requests admitted together share an index even if their
total step counts differ. After prefix-length grouping, each transformer forward
requires a common index. Mixed indices within a group raise a `ValueError`
before step publication, including in native precision. On exit the pipeline restores the
previous index directly, preserving the worker's paged-KV activation scope.

The explicit Triton provider accepts H3 tile64/128 on SM80+ in BF16.
For tile128 it expands the selected mask into 64-token compute blocks
without recomputing top-k or splitting the softmax reduction. Ada tile128
auto selection uses this provider. Explicit unsupported precision and
missing-kernel errors remain startup failures, not runtime fallbacks.

Use a new backend for a new provider/capability contract. `custom_attention`
instead owns communication and requires `skip_sequence_parallel=True`.

## Adding or changing a backend

An implementation change is complete only when it:

1. implements the `AttentionBackend` interface and exposes a stable enum name;
2. adds platform validation and default routing where appropriate;
3. defines and validates any typed configuration;
4. tests explicit selection, default selection, and incompatible paths;
5. documents installation, fallback behavior, and quality implications in the
   corresponding user guide; and
6. updates the overview matrix without moving algorithm details back into the
   landing page.

## FastVideo VSA model metadata

FastWan's text-to-video and image-to-video modes both use `Wan22Pipeline`.
It operates on a flattened DiT sequence but partitions tokens in the original
latent video grid. Wan integrations therefore attach the
post-patch `(T, H, W)` shape as `vsa_dit_seq_shape` attention metadata. The
backend validates that `T * H * W` equals the sequence length, derives the
runtime block count, and selects the configured top-k key/value blocks for
each query block.

Checkpoint-specific compensation remains model-owned. Wan layers pass an
optional learned `gate_compress` projection through attention metadata when
the checkpoint contains those weights. Checkpoints without the projection do
not allocate or execute it. When top-k selects every block, native checkpoints
route to SDPA, while FastVideo DMD checkpoints preserve the VSA all-block path
to retain checkpoint semantics.

MiniMax-H3 supplies `MiniMaxH3VSAImpl` through `impl_overrides`.
`models/minimax_h3/attention/fastvideo_h3.py` owns its prefix layout, video tiling, and
learned compression gate. Pooling, block-map construction, and tile64 provider
calls are shared through `attention/ops/block_sparse.py`.

## CosyVoice3 Ascend graph region

CosyVoice3 has an opt-in, model-owned NPU graph adapter around the complete
Flow Euler solve. The adapter directly calls the FIA inference API for dense,
noncausal FP16/BF16 DiT attention with a boolean key-padding mask. It converts
keep-mask semantics to FIA's blocked-mask semantics and preserves masking of
padded query outputs in the model. FIA also serves the adapter's eager warmup
and graph-limit fallback paths. Selecting another backend does not replace FIA
while the adapter is enabled. Disable the adapter to exercise the generic
backend path. The ordinary selector and other models are unchanged.
See the [CosyVoice3 serving guide](../../user_guide/examples/online_serving/text_to_speech.md#ascend-stage-1-graph-regions)
for enablement, supported dtypes, capture limits, and failure behavior.
