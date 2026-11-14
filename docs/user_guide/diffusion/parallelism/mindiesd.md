# MiniMax-H3 MindIE-SD parallel attention

MindIE-SD owns Q/K/V prepacking, forward/reverse all-to-all, RF-v3 pooling/mask
preparation, CANN BF16 BSA and communication/compute overlap. Omni supplies
request metadata, admission and state lifetime; admitted calls bypass native
Ulysses communication. This integration enables parallelism and RF reorder
hoist, without SDK fusion operators, NZ weights, Eagle BSA or head-major QKV.

## Requirements and configuration

Use a MindIE-SD `dev` build exposing `mindiesd.build_rf_attention_plan` and
`mindiesd.parallel.{ParallelState, distributed_sparse_attention}`, with matching
native plugin/OPP. Admission requires Ascend NPU, BF16 RF-v3, equal `[1,S,H,D]`
Q/K/V, strict pure Ulysses and heads divisible by its degree. Ring, AllGather-KV,
paged/reference KV, joint attention, VDN and gate compression are unsupported.
RF reorder hoist also requires every main-DiT layer to use compatible settings,
no attention schedule, and one request per forward.

Enable the executor through `block_sparse.parallel`:

```json
{
  "default": {
    "backend": "RAINFUSION_ATTN",
    "block_sparse": {
      "sparsity": 0.8,
      "start_step": 0,
      "precision": "bf16",
      "sparse_type": "rf_v3",
      "parallel": {"overlap_comm": true, "reorder_hoist": true}
    }
  }
}
```

Omit `parallel` to retain ordinary Omni attention. `overlap_comm=false` selects
the SDK serial schedule for diagnosis. `reorder_hoist` builds the RF plan once
before RoPE, relabels text/image/audio/output positions and lets SD skip per-layer
row permutation while preserving RF order. Sparse step/layer gates still apply.

For the validated AICPU communication mode, set
`VLLM_OMNI_ULYSSES_HCCL_EXPANSION_MODE=2` before startup. It selects the HCCL
engine only for Ulysses groups; verify all participating ranks' group logs.

## Execution and state ownership

Unsupported original-order calls decline before SDK state allocation or
communication. A rejected RF-reordered sparse call raises to prevent an unaware
fallback. Execution failures poison the shared resource and propagate without
collective retry.

The SP coordinator owns one `ParallelState` per Ulysses group/device, reusing
SDK mappings, chains, receive buffers, stream and events across layers/steps.
Calls must be serialized and share the same overlap mode. Before group teardown,
Omni synchronizes the device and releases state; the SDK has no public close API.

Configuration lives beside `BlockSparseSpec` in `data.py`; executor resources
live with their owner in `group_coordinator.py`. The two H3 attention helpers
handle per-layer SDK execution and request-level reorder, sharing one Ulysses
topology check. Additional SDK capabilities can extend these boundaries.

## Validation

CPU contracts cover admission, configuration, state reuse/failure/teardown and
row labels. Native A/B tests must match source, SDK build, model, input,
precision, sparsity, NFE and cards, with two complete warmups before formal
timing. Record HTTP/media completion, actual NFE, SDK engagement, Diffuse/E2E
latency and measured peak memory. CPU tests do not establish native numerics;
fusion/NZ measurements do not establish this parallel-only path's performance.
