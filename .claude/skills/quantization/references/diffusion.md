# Diffusion Quantization Reference

Use this file for DiT quantization, config routing, checkpoint adapters, and
quality or loader debugging.

## Supported Method Families

The current local diffusion quantization families are:

- CUDA/GPU online or checkpoint: `fp8`, `int8`, `gguf`, ModelOpt
- Ascend NPU: `int8`, `mxfp8`, `mxfp4`, `mxfp4_dualscale`, msModelSlim paths
- Intel XPU: AutoRound/INC and online MXFP8 paths where documented

Always cross-check the method doc before claiming model support. The CLI
accepting a method string is not enough.

## Implementation Starting Points

| Area | Files |
|------|-------|
| Unified factory | `vllm_omni/quantization/factory.py` |
| Per-component routing | `vllm_omni/quantization/component_config.py` |
| GGUF diffusion method | `vllm_omni/quantization/gguf_config.py` |
| Int8 diffusion method | `vllm_omni/quantization/int8_config.py` |
| MXFP8 diffusion method | `vllm_omni/quantization/mxfp8_config.py` |
| MXFP4 and dual-scale method | `vllm_omni/quantization/mxfp4_config.py` |
| AutoRound/INC omni extension | `vllm_omni/quantization/inc_config.py` |
| ModelOpt checkpoint adapters | `vllm_omni/diffusion/model_loader/checkpoint_adapters/modelopt.py` |
| FP8 scaled-mm runtime patches | `vllm_omni/patch.py` |
| Similarity tool | `vllm_omni/quantization/tools/compare_diffusion_trajectory_similarity.py` |

## Threading `quant_config`

For a diffusion transformer:

1. Accept `quant_config: QuantizationConfig | None` in the transformer.
2. Pass `quant_config` and stable `prefix` values to every vLLM linear layer.
3. Preserve packed-module mappings for fused QKV, FFN, or model-specific fused
   projections.
4. Call `apply_vllm_mapper()` on configs that need HF-to-vLLM name remapping.
5. For component configs, keep prefixes specific enough to avoid quantizing VAE
   or encoders by accident.

Common skipped areas are `proj_out`, output projections, modulation layers,
embedders, refiner blocks, MoE routers, and layernorm-adjacent paths.

## Online vs Offline

Online quantization:

- starts from BF16/FP16 checkpoint weights
- computes or stores quantized weights at load time
- typically uses `--quantization <method>` or `quantization="<method>"`

Offline or pre-quantized checkpoints:

- include quantized weights and scales on disk
- usually carry `quantization_config` in `transformer/config.json`
- should auto-detect when the config is complete
- may require model-specific adapter logic for tensor names and packed modules

Do not mix online flags with incompatible offline checkpoint configs. If the
active config and disk config differ, `resolve_quant_config_from_disk()` should
raise or rebuild rather than silently loading corrupted weights.

## GGUF Rules

- Require `quantization_config.gguf_model`.
- Use the normal base model for tokenizer, scheduler, text encoder, and VAE.
- Add explicit architecture adapters for tensor-name mapping.
- Use `named_parameters()` and `named_buffers()`, not `state_dict()`, to
  discover loadable names.
- Guard fused QKV/KV rewrites so tensors are not rewritten twice.
- Flatten N-D activations around the GGUF matmul and restore shape.

Unsupported architectures should fail clearly instead of falling back to a fake
generic mapper.

## MXFP Rules

- `mxfp8` is W8A8 microscaling, currently centered on NPU and documented XPU
  behavior.
- `mxfp4` is online single-scale W4A4 on NPU.
- `mxfp4_dualscale` is online/offline dual-scale W4A4 with BF16 fallback; the
  offline path is the production path when calibrated `mul_scale` exists.
- Offline dual-scale checkpoints must inject per-transformer `ignored_layers`
  into config because cascade transformers can have different BF16 fallback
  sets.

Do not load an offline dual-scale checkpoint with the online single-scale path.

## Common Failures

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `Unknown quantization method` | Factory override or upstream registration is missing | Add the right override or upstream method |
| Missing scale tensors | Wrong scope, wrong adapter, or BF16 layer not allowed to dequantize | Check config scope and adapter scale routing |
| Output rank collapses from 3-D to 2-D | FP8 scaled-mm kernel ignored output shape | Check the patch in `vllm_omni/patch.py` |
| Quality degrades silently | Sensitive layers quantized | Add `ignored_layers`; compare to BF16 |
| Only one cascade transformer works | Disk config differs per transformer but active config is reused | Rebuild from each transformer's config |
| Quantized path slower than BF16 | Kernel, dtype, compile, or benchmark mode mismatch | Compare eager-to-eager and non-eager-to-non-eager |

## Verification

For a new or changed diffusion quantization path, run:

1. Config and loader smoke test.
2. One fixed-seed generation against BF16.
3. Numeric comparison with the similarity tool or equivalent metrics.
4. Serving test if the path is user-facing online.
5. Memory and latency check under the same execution mode as the baseline.

Keep output artifacts when precision changes are part of the PR.
