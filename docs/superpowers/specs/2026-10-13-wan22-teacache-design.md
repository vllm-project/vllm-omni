# Wan 2.2 TeaCache distributed-support design

## Goal

Make TeaCache for `WanTransformer3DModel` behaviorally equivalent to the
native Wan 2.2 forward path for single-device execution, pipeline parallelism
(PP), and sequence parallelism (SP). A cache miss and a cache hit must preserve
the native output contract.

## Scope

- Preserve native AdaLayerNorm calls: `norm1(x, scale_msa, shift_msa)` and
  `norm_out(x, scale, shift)`.
- Preserve T2V, I2V, and TI2V conditioning and output behavior.
- Preserve PP ownership: first-stage patch embedding, stage-local block ranges,
  intermediate-tensor handoff, and last-stage output projection.
- Preserve SP sharding, padding masking, VSA geometry, and stage-local cache
  residuals.
- Add tests for single-device cache miss/hit equivalence and focused PP/SP
  routing semantics.

## Design

`extract_wan2_2_context` will mirror the native Wan forward preparation path
up to the first local transformer block. On first PP stages it will patchify
and invoke `_sp_shard_point`; later PP stages will consume `hidden_states` from
`IntermediateTensors`. Conditioning, RoPE, timestep preparation, local SP
padding masks, and VSA settings will follow the native implementation.

The cache context's modulated input is computed with the first local block's
native AdaLayerNorm call. Its transformer closure runs only
`blocks[start_layer:end_layer]`, passing the same attention-mask and VSA
arguments as native forward. The postprocess closure returns
`IntermediateTensors` on non-final PP stages; on the final stage it applies the
native output AdaLayerNorm, projection, unpatchification, and `return_dict`
contract.

Cache state is not moved across PP or SP boundaries: residuals remain local to
the owning stage and sequence shard. This matches the tensors used by the
corresponding native block range and avoids mixing full-sequence and sharded
representations.

## Validation

- Unit-test T2V, I2V, and TI2V cache miss/hit outputs against the native
  forward contract.
- Mock PP first/middle/last stages and assert patching, local block slicing,
  `IntermediateTensors`, and final projection ownership.
- Mock SP sharding/padding and assert the local mask, VSA grid, and block
  arguments are preserved.
- Retain the documentation feature flag only after these tests pass.
