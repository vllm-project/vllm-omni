# HF Numerics for RL Rollouts (Qwen3-Omni Thinker)

`hf_numerics` is an opt-in mode of the Qwen3-Omni thinker stage for true on-policy
RL. With it enabled, the text decoder computes RMSNorm, the residual and deepstack adds, rotary
embedding and MoE routing exactly as the transformers implementation does, and runs the
experts through vLLM's `fused_experts`, which the trainer registers as the experts
implementation of the HF model. A trainer that runs the HF model under batch-invariant
kernels then recomputes the rollout log-probs bitwise instead of within a tolerance.

## What changes

| Op | Default | `hf_numerics` |
| --- | --- | --- |
| RMSNorm (input, post-attention, q/k, final) | fused kernel, weight multiplied in fp32 | normalised in fp32, cast to the activation dtype, then multiplied by the weight |
| Residual add | fused add-norm on the unrounded sum | plain add in the activation dtype, norm on the rounded sum |
| Deepstack visual add | added to the MLP output before the residual add | added to the rounded layer output (residual + MLP) |
| Rotary embedding | vLLM M-RoPE kernel | HF cos/sin computed in fp32 per call and the HF `rotate_half` formula |
| MoE routing | `fused_topk` | fp32 softmax, top-k, renormalise, cast to the activation dtype, as the HF router |
| MoE experts | `fused_experts` | unchanged; the trainer side runs the same kernel through transformers' `ExpertsInterface` |

Attention, dense GEMMs, embedding and the LM head are unchanged: they already match
the HF model bitwise under `VLLM_BATCH_INVARIANT=1`, as long as both sides use the
same FlashAttention kernel. `fused_experts` is batch-invariant under the same flag, so
the trainer's packed batch and the rollout's decode batch give the same rows.

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
supported. One more requirement is checked when the model is built: the
**unquantized Triton MoE backend**. The MoE block reads the loaded expert weights
as HF `[experts, 2 * intermediate, hidden]` / `[experts, hidden, intermediate]`
matrices. The FlashInfer backends repack them after loading, so any other backend
is rejected; set `moe_backend: triton` (`--moe-backend triton`) when `auto` would
pick FlashInfer.

## Cost

The replaced norm, rotary and routing ops are unfused eager PyTorch; the experts keep the
fused kernel. Keep the mode off for serving.
