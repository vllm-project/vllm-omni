# MiniMax-H3 reference optimizations

These opt-in controls add NPU-resident reference KV and independent image
sparsity to Ref2VA. They can be used with existing compatible adapters and
sampling settings; neither requires the fixed eight-step LQ-Slim preset.
See [LQ-Slim](../lq_slim/README.md) for the complete install, serve and request example.

## Reference KV reuse

Append the existing configuration flag to your normal Ascend Ref2VA server:

```bash
--kv-offload-config '{"mode":"on","kv_storage":"npu","kv_refresh_interval":8,"kv_host_quantization":"none"}'
```

Interval 8 refreshes at steps 0, 8, 16 and so on. The default `host` storage
keeps layerwise prefetch; `npu` keeps unquantized KV on the device and avoids
host copies at the cost of additional resident device memory. Cache precision
is independent of DiT weight and attention quantization. The cache belongs to
one request; model weights and the adapter must remain fixed until it closes.

No SP and pure Ulysses are supported; Ring and paged KV are unsupported.
Transport padding is removed before attention and reference KV capture.

## Independent image sparsity

Keep normal `block_sparse` settings and add a RainFusion-specific image deletion ratio:

```bash
--diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix"},"rainfusion":{"image_sparsity":0.9}}}'
```

The ratio is in `[0, 1]`; default 0 disables image selection. Per image, head and
eligible video query block, the backend keeps at least one candidate block,
using `max(1, ceil(N * (1 - image_sparsity)))`. Reference-video queries participate
alongside target-video queries; protected queries and image boundary blocks
retain their protection. Image selection is independent of global sparsity and
does not reduce the reference KV allocation.

This requires the public MindIE-SD `image_spans/image_sparsity` API and matching
rf_v2/rf_v3 implementations. The checked source is
[MindIE-SD `6965ff2`](https://gitcode.com/Ascend/MindIE-SD/tree/6965ff2f397c3370ef50d06b6564067d3205fccb).
The old RBS helper patch is insufficient. Install the dependency using its
normal instructions; Omni rejects an incompatible public API before execution.

MiniMax-H3 supplies independent image geometry through packing and cache
compaction. On reuse steps with selected images, dummy reference queries are
added after Ulysses and discarded after attention to satisfy the equal-length
Q/K/V ABI. Reference projections and KV stay cached. With image selection off,
A5 resident reuse retains rectangular BF16/mix execution with unquantized KV.

Use the existing [attention schedule](../../docs/user_guide/diffusion/attention_backends/schedules.md)
to vary video sparsity by step. Each profile is a complete configuration and
must repeat its image option. The normal request and sampling interfaces apply.

## Validation

Focused CPU coverage checks configuration, image geometry, default routing,
step profiles, host/resident KV values and cleanup, and SP alignment. The
optional dependency-source tests execute real Python permutation/pooling/mask
functions from the checked MindIE-SD source. Native calls are spies; these tests
do not validate device operators or distributed execution.

The main port has not been validated end to end on NPU. Performance and output
quality remain unverified for this revision. The complete recipe includes
CPU prerequisites and local/CI test commands.
