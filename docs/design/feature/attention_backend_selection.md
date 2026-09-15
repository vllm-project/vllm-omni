# Diffusion Attention Backend Selection

This document defines the selection and extension contract for diffusion
attention backends. User-facing backend choices, installation, and tuning are
in the [attention backend guides](../../user_guide/diffusion/attention_backends.md).

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

The selector must not duplicate device capability or package-availability
policy that belongs to the platform.

## Typed backend options

Backend-specific settings remain typed fields on `AttentionSpec` rather than
unstructured keyword dictionaries:

- `quant` is consumed by FlashInfer and TRTLLM with backend-specific value
  validation;
- `aiter_quant` is consumed by AITER MHA v4 and selects an
  architecture-compatible operand format;
- `skip_softmax` is consumed by TRTLLM; and
- `block_sparse` is consumed by block-sparse backends such as RainFusion; and
- `fastvideo_vsa_topk` is consumed by FastVideo VSA.

A backend reads only the fields it owns and rejects incompatible values. New
options should be added to a shared typed spec only when more than one backend
shares their semantics; otherwise add a dedicated typed block.

## Model integration contract

Each diffusion attention call declares a stable role and, when appropriate, a
broader category. Model-specific roles permit precise overrides; categories
preserve common `self` or `cross` policies. A model also decides whether its
path is compatible with automatic TRTLLM selection, because only the model
knows whether masking and packing satisfy the kernel contract.

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

FastVideo VSA support is scoped to `FastVideo/FastWan2.2-TI2V-5B-Diffusers`, whose text-to-video and image-to-video modes both use `Wan22Pipeline`. It operates on a flattened DiT sequence but partitions tokens in
the original latent video grid. Wan integrations therefore attach the
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
