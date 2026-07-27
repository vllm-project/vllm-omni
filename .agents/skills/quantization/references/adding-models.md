# Adding Quantization Support to a Model

Use this checklist when extending quantization to a new architecture or stage.

## Classify Ownership

| Case | Owner |
|------|-------|
| New generic AR method, kernel, or loader semantics | upstream `vllm` first |
| Existing upstream method needs diffusion wiring | `vllm-omni` model and config integration |
| GGUF or ModelOpt tensor mapping for a diffusion architecture | `vllm-omni` adapter |
| NPU MXFP or msModelSlim diffusion path | `vllm-omni` platform/method wrapper and docs |
| Multi-stage omni checkpoint quantizes only the thinker LM | `vllm-omni` component scoping |

Avoid building a private quantization stack in model code.

## Model Wiring Checklist

1. Find the closest supported model implementation.
2. Confirm the model uses vLLM linear layers that accept `quant_config`.
3. Add `quant_config` to the transformer constructor.
4. Pass stable `prefix` values through every linear layer.
5. Handle fused modules and packed mappings explicitly.
6. Ensure `load_weights()` and any `WeightsMapper` remap `ignored_layers` or
   `modules_to_not_convert`.
7. Keep tokenizer, scheduler, VAE, text encoder, audio/vision encoders, and
   talker BF16 unless method-specific docs validate them.
8. Add or update a method doc under `docs/user_guide/quantization/`.

## Checkpoint Adapter Checklist

For pre-quantized checkpoints:

- Inspect `transformer/config.json` first.
- Check `quant_method`, `method`, `producer.name`, `quant_algo`, serialized
  flags, `ignored_layers`, and any method-specific fields.
- Verify tensor name mapping before generation.
- Verify packed modules such as `to_qkv`, `add_kv_proj`, and `w13`.
- Confirm scale tensor names and shapes.
- Allow dequantization back to BF16 for intentionally skipped target layers.

## Sensitive Layer Discovery

Start conservative. Quantize the backbone first and keep output-side or routing
layers BF16 until quality is proven.

Frequently sensitive areas:

- attention output projections such as `to_out`
- FFN down projections such as `w2`
- final projections such as `proj_out`, `final_layer`, and output heads
- modulation, timestep, position, image, or text embedders
- refiner blocks
- MoE routers, experts selected by routing, and layernorm-adjacent modules
- VAE, text/vision towers, audio encoders, and codec/talker stages

Do not reuse one model family's ignore list blindly.

## Tests and Evidence

Minimum evidence for a PR:

- loader/config test or smoke command
- fixed-seed BF16 vs quantized output comparison
- quality metrics and saved outputs for precision-changing work
- memory and latency comparison under identical execution mode
- failure behavior for unsupported models or missing checkpoint fields

When the PR adds a method to a support table, include the exact model,
hardware, command, quality threshold, and artifact path used for validation.
