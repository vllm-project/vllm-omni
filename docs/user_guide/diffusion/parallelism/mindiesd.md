# MiniMax-H3 MindIE-SD parallel attention

## Requirements and configuration

Use a matching MindIE-SD dev build and native plugin/OPP exposing `mindiesd.build_rf_attention_plan` and `mindiesd.parallel.{ParallelState, distributed_sparse_attention}`:

| Path | `sparse_type` | `precision` | `bsa_backend` | Communication | Hardware |
| --- | --- | --- | --- | --- | --- |
| RainFusionAttention | `rf_v2` | `bf16` / `fp16` | `cann` | BF16 / FP16 Q/K/V | A2 / A3 |
| Eagle BSA | `rf_v3` | `bf16` / `fp16` | `eagle` | BF16 / FP16 Q/K/V | A2 / A3 |
| Eagle QBSA | `rf_v3` | `mix` | `eagle` | BF16; optionally INT8 Q/K, FP8 V | A5 |
| Native MXFP8 BSA | `rf_v3` | `mxfp8` | `cann` | BF16; optionally MXFP8 Q/K/V and E8M0 scales | A5 |

Admission requires Ascend NPU, equal `[1,S_local,H,D]` Q/K/V, strict Ulysses and divisible heads.
CP x Ulysses requires `rainfusion_reorder_hoist=true` (`pre_rearranged=True` at runtime), U4/U8, CP2/CP4, sequence world 8/16/32 and D128. Pure-Ulysses MXFP8 also requires D128.
Ring, paged/reference KV, joint attention, VDN and gate compression are unsupported. RF hoisting requires compatible settings on all main-DiT layers, no attention schedule and one request per forward.

```json
{"default": {"backend": "RAINFUSION_ATTN", "block_sparse": {
  "sparsity": 0.8, "precision": "bf16", "sparse_type": "rf_v3", "parallel": {}
}}}
```

Omit `parallel` for ordinary Omni attention; set `overlap_comm=false` for the SDK serial path.
`optimize_attention_pipeline` reuses SDK preparation/buffers. Hoisting relabels rows and timestep caches before sharding, independently of sparse-step gating.
For mix/MXFP8, `quantized_comm=false` defaults to BF16 transfers; set it to `true` for quantized transfers consistently on all workers.

Save the JSON as `mindiesd-attention.json`:

```bash
VLLM_LOGGING_LEVEL=DEBUG MINDIE_LOG_LEVEL=sd:debug \
  vllm serve /path/to/MiniMax-H3 --omni --trust-remote-code \
    --task-type fl2va --num-gpus 8 --usp 8 --ring 1 \
    --diffusion-attention-config "$(cat mindiesd-attention.json)"
```

Submit a request reaching a sparse step; check worker logs for SDK `[parallel] rank=... world=... mode=...`.
See the [design and execution flow](../../../design/feature/attention_backend_selection.md#minimax-h3-mindie-sd) and [benchmark guidance](../../../design/feature/attention_backend_selection.md#mindie-sd-benchmarking).
