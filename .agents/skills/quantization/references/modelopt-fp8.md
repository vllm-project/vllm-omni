# ModelOpt FP8 Conversion and Loading

Use this when exporting or loading ModelOpt FP8 checkpoints for diffusion
transformers. This reference captures the practical conversion lessons from the
2026-04-21 ModelOpt FP8 paste and aligns them with the current
`vllm-omni` ModelOpt adapter.

## Core Rule

Treat ModelOpt FP8 as selective checkpoint quantization, not one-click
whole-model FP8. Quantize the main transformer or DiT backbone, keep sensitive
or peripheral modules BF16, serialize a complete ModelOpt checkpoint, and let
vLLM-Omni auto-detect it from config.

Do not pass `--quantization fp8` to force a ModelOpt checkpoint path. The
checkpoint's `quantization_config` should select `modelopt`.

## Conversion Strategy

Prefer:

- backbone attention Q/K/V and FFN expansion projections
- conservative per-model ignore lists
- MHA quantizers disabled until basic linear quantization is stable
- full module-name prefixes in ignore/routing logic
- BF16 fallback for output-side and routing-sensitive layers

Avoid quantizing by default:

- VAE
- text, vision, or audio encoders
- embedders and modulation layers
- final heads and output projections
- refiner modules
- MoE routers and layernorm-sensitive blocks

## Model-Specific Starting Points

| Model Family | Quantize First | Keep BF16 First |
|--------------|----------------|-----------------|
| Z-Image | attention q/k/v, FFN w1/w3 | `to_out`, `w2`, `noise_refiner`, `context_refiner`, embedders, modulation, final/output |
| Qwen-Image | transformer backbone linear layers | `proj_out`, `img_in`, `txt_in`, norms, modulation, position/time/embed layers |
| FLUX2-dev / FLUX2-klein | backbone linear layers | `proj_out`, context/x embedders, `norm_out`, stream modulation, time embeddings |
| HunyuanImage-3.0 | `model.model` backbone | VAE, vision tower, token/time/image embedders, final image head, LM head, routers, layernorm-sensitive blocks |

These are starting points, not support claims. Validate with BF16 outputs.

## Required Checkpoint Metadata

`transformer/config.json` must include a complete `quantization_config`.
At minimum it should identify ModelOpt and the algorithm:

```json
{
  "quantization_config": {
    "producer": {"name": "modelopt"},
    "quant_method": "modelopt",
    "quant_algo": "FP8"
  }
}
```

Current vLLM-Omni auto-detects:

- `FP8` and `FP8_PER_CHANNEL_PER_TOKEN` -> `modelopt`
- `NVFP4` -> `modelopt_fp4`
- `MIXED_PRECISION` -> `modelopt_mixed`

Packed modules and scales must map correctly:

- `to_qkv` from `to_q`, `to_k`, `to_v`
- `add_kv_proj` from `add_q_proj`, `add_k_proj`, `add_v_proj`
- `w13` from `w1`, `w3`
- `.input_scale`, `.weight_scale`, `.weight_scale_2`, `.weight_scale_inv`

For selective FP8, target layers that remain BF16 must be allowed to
dequantize from FP8 weights using the matching `weight_scale`, or the adapter
must skip the quantized scale tensors for that BF16 target.

## Runtime Signals

Healthy loading should show ModelOpt auto-detection and a ModelOpt linear
kernel path. Useful signals include:

- auto-detected quantization from config
- selected ModelOpt FP8 linear method
- CUTLASS or FlashInfer FP8 scaled-mm backend when enabled and supported
- non-eager path compiling the transformer when expected

If online serving fails while offline inference works, inspect warmup and
serving initialization before changing checkpoint conversion.

## Validation Order

1. Check config completeness.
2. Run an export checker such as `examples/quantization/check_modelopt_fp8_export.py`.
3. Confirm actual FP8 tensors and expected disk-size reduction.
4. Run one offline generation with fixed prompt, seed, size, and steps.
5. Run serving with `vllm serve --omni` and verify `/v1/images/generations`.
6. Compare against BF16 using visual review plus metrics such as LPIPS, PSNR,
   MAE, cosine similarity, and SSIM.
7. Compare performance only under matched modes:
   - BF16 eager vs FP8 eager
   - BF16 non-eager vs FP8 non-eager

FP8 is not automatically faster. Check whether execution reaches FP8 kernels,
whether compile mode is comparable, and whether the workload shape fits the
kernel path.

## Triage

| Symptom | First Check |
|---------|-------------|
| Startup failure | `config.json`, `model_index.json`, `quantization_config`, missing scales |
| Loads but output is corrupted | Over-quantized sensitive layers or wrong full-name prefixes |
| Offline works but online fails | Dummy warmup, serving init, compile path |
| FP8 is not faster | Kernel selection, dtype, compile mode, workload shape |
| BF16 fallback fails | Missing `weight_scale` for dequantization or wrong packed-module mapping |

The success criterion is stable, correctly scoped checkpoint loading with
quality preserved enough for the target use case, not maximizing the number of
FP8 layers.
