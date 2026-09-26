# HF Numerics for RL Rollouts (Qwen3-Omni Thinker)

`hf_numerics` is an opt-in mode of the Qwen3-Omni thinker stage for true on-policy
RL. With it enabled, the text decoder computes RMSNorm, the residual and deepstack adds, rotary
embedding and the MoE block exactly as the transformers implementation does, so a
trainer that runs the HF model under batch-invariant kernels recomputes the rollout
log-probs bitwise instead of within a tolerance.

## What changes

| Op | Default | `hf_numerics` |
| --- | --- | --- |
| RMSNorm (input, post-attention, q/k, final) | fused kernel, weight multiplied in fp32 | normalised in fp32, cast to the activation dtype, then multiplied by the weight |
| Residual add | fused add-norm on the unrounded sum | plain add in the activation dtype, norm on the rounded sum |
| Deepstack visual add | added to the MLP output before the residual add | added to the rounded layer output (residual + MLP) |
| Rotary embedding | vLLM M-RoPE kernel | HF cos/sin computed in fp32 per call and the HF `rotate_half` formula |
| MoE | `fused_topk` + fused experts | fp32 softmax, top-k, renormalise, cast to the activation dtype, then `grouped_mm` experts |

Attention, dense GEMMs, embedding and the LM head are unchanged: they already match
the HF model bitwise under `VLLM_BATCH_INVARIANT=1`, as long as both sides use the
same FlashAttention kernel.

Each replaced op is a `torch.library` custom op, so it is opaque to `torch.compile`.

## Enabling it

Set it in the thinker stage's engine extras of the deploy configuration:

```yaml
stages:
  - stage_id: 0
    engine_extras:
      additional_config:
        hf_numerics: true
```

The mode is meant to be combined with `VLLM_BATCH_INVARIANT=1`, `tensor_parallel_size: 1`
and no expert parallelism. Only the default rope type and MoE decoder layers are
supported. Two more requirements are checked when the model is built:

- **Unquantized Triton MoE backend.** The MoE block reads the loaded expert weights
  as HF `[experts, 2 * intermediate, hidden]` / `[experts, hidden, intermediate]`
  matrices. The FlashInfer backends repack them after loading, so any other backend
  is rejected; set `moe_backend: triton` (`--moe-backend triton`) when `auto` would
  pick FlashInfer.
- **CUDA graphs only with a native `grouped_mm` kernel.** In PyTorch 2.13,
  `torch._grouped_mm` has a native kernel only for bf16 on SM90 (Hopper) and SM100
  (datacenter Blackwell); see `_grouped_mm_cuda` in
  `aten/src/ATen/native/cuda/GroupedBlas.cpp`. Everywhere else it falls back to a per-expert loop that
  copies the group offsets to the host, which cannot be captured into a CUDA graph.
  On other GPUs or dtypes, set `enforce_eager: true`.

## Cost

The replaced ops are unfused eager PyTorch. On a 4-layer Qwen3-Omni-30B-A3B thinker on
one H20, thinker-only stage with CUDA graphs, 32 concurrent requests × 512 tokens,
throughput went from 3318 to 2228 tok/s. Keep the mode off for serving.
