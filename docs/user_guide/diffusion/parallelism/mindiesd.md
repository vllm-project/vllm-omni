# MiniMax-H3 MindIE-SD parallel attention

Omni delegates complete parallel attention to MindIE-SD and retains request layout, admission and State lifetime. Enable it with `block_sparse.parallel`.

## Requirements and configuration

Use a matching MindIE-SD `lly/minimax-h3-u8-kvgather2-20261110` build and native plugin/OPP.
It must expose `mindiesd.build_rf_attention_plan` and `mindiesd.parallel.{ParallelState, distributed_sparse_attention}`:

| Path | `sparse_type` | `precision` | `bsa_backend` | Communication | Hardware |
| --- | --- | --- | --- | --- | --- |
| RainFusionAttention | `rf_v2` | `bf16` / `fp16` | `cann` | BF16 / FP16 Q/K/V | A2 / A3 |
| Eagle BSA | `rf_v3` | `bf16` / `fp16` | `eagle` | BF16 / FP16 Q/K/V | A2 / A3 |
| Eagle QBSA | `rf_v3` | `mix` | `eagle` | BF16; optionally INT8 Q/K, FP8 V | A5 |
| Native MXFP8 BSA | `rf_v3` | `mxfp8` | `cann` | BF16; optionally MXFP8 Q/K/V and E8M0 scales | A5 |

`bsa_backend` defaults to the table's provider; conflicting providers or algorithm/precision tuples fail configuration.
Admission requires Ascend NPU, equal `[1,S_local,H,D]` Q/K/V, strict Ulysses and
heads divisible by its degree. FP16 converts BF16 producer operands explicitly
and restores the producer dtype; quantized paths consume BF16 producer tensors.
Pure Ulysses sets `cp_group=None` in `ParallelState` and accepts either RF input order.
CP x Ulysses uses the existing `ulysses_allgather_kv` strategy and supplies the
coordinator's orthogonal `allgather_group` as `ParallelState.cp_group`. It requires
`pre_rearranged=True`, Ulysses degree 4 or 8, CP degree 2 or 4, global sequence
world 8, 16 or 32, and D128. MXFP8 also requires D128 in pure Ulysses.
Ring, paged/reference KV, joint attention, VDN and gate compression are unsupported. RF hoisting requires compatible settings on all main-DiT layers, no attention schedule and one request per forward.

```json
{
  "default": {
    "backend": "RAINFUSION_ATTN",
    "block_sparse": {
      "sparsity": 0.8,
      "precision": "bf16",
      "sparse_type": "rf_v3",
      "parallel": {
        "overlap_comm": true,
        "rainfusion_reorder_hoist": true,
        "optimize_attention_pipeline": true
      }
    }
  }
}
```

Omit `parallel` for ordinary Omni attention; set `overlap_comm=false` for the SDK serial path.
`optimize_attention_pipeline` enables SDK preparation/buffer reuse. Hoisting
relabels tokens before sharding, independently of when sparse steps start.
For mix and MXFP8, `quantized_comm` defaults to `false` (BF16 transfers).
Set it to `true` for quantized transfers; all workers must use the same setting.

## Launch and verify

Save the JSON above as `mindiesd-attention.json`. Install the compatible SDK wheel
in the same environment as Omni, and load its native plugin and OPP environment
before starting the server:

```bash
VLLM_LOGGING_LEVEL=DEBUG MINDIE_LOG_LEVEL=sd:debug \
  vllm serve /path/to/MiniMax-H3 --omni --trust-remote-code \
    --task-type fl2va --num-gpus 8 --usp 8 --ring 1 \
    --diffusion-attention-config "$(cat mindiesd-attention.json)"
```

Submit a generation request that reaches a sparse denoising step. Check worker
logs for the SDK `[parallel] rank=... world=... mode=...` message.

See the [design and execution flow](../../../design/feature/attention_backend_selection.md#minimax-h3-mindie-sd)
and [benchmark guidance](../../../design/feature/attention_backend_selection.md#mindie-sd-benchmarking).
