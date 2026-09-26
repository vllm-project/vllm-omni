---
name: quantization
description: Work on vLLM-Omni quantization for diffusion, autoregressive, omni, or multi-stage models. Use when choosing or adding methods such as fp8, int8, gguf, mxfp8, mxfp4, mxfp4_dualscale, ModelOpt, AutoRound, INC, msModelSlim, awq, or gptq; debugging quantized loading; or validating memory, speed, and output quality.
---

# vLLM-Omni Quantization

Use this skill for quantization work in `vllm-omni`. Start from the local
code and docs, then pick the smallest validated path for the model, method,
hardware, and quality target.

## First Checks

Before changing code or recommending a command, identify:

- Model and task: AR text, omni/TTS, text-to-image, text-to-video, or multi-stage diffusion.
- Hardware backend: CUDA generation, Ascend NPU, Intel XPU, or another platform.
- Quantization mode: online from BF16/FP16 weights, pre-quantized checkpoint, runtime KV/FA quantization, or per-component routing.
- Scope: transformer/DiT, thinker language model, text encoder, VAE, audio/vision encoder, talker, or a specific stage.
- Validation target: memory reduction, latency/throughput, output quality, or all three.

Do not assume a method is supported just because the CLI accepts the string.
Check `docs/user_guide/quantization/` and the closest implementation first.

## Quick Decision

| Task | Start With |
|------|------------|
| Choose a method or command | `references/methods.md` and `references/modality-compat.md` |
| Use `build_quant_config()` or per-component routing | `references/methods.md` |
| Work on diffusion quantization | `references/diffusion.md` |
| Add quantization to a new model | `references/adding-models.md` |
| Convert or load ModelOpt FP8 checkpoints | `references/modelopt-fp8.md` |
| Debug ModelOpt, GGUF, AutoRound, MXFP, or serialized Int8 loading | `references/diffusion.md` and method docs under `docs/user_guide/quantization/` |

## Current Runtime Shape

The unified entrypoint is:

```python
from vllm_omni.quantization import build_quant_config
```

It supports method strings, flat method dictionaries, per-component
dictionaries, existing `QuantizationConfig` objects, and `None`. The factory
delegates generic methods to upstream `vllm` and keeps vLLM-Omni overrides for
diffusion or omni-specific routing:

- `gguf`
- `int8`
- `mxfp8`
- `mxfp4`
- `mxfp4_dualscale`
- `inc`, `auto-round`, `auto_round`
- ModelOpt auto-detected configs: `modelopt`, `modelopt_fp4`, `modelopt_mixed`

Use `ComponentQuantizationConfig` when only one stage or component should be
quantized. Pre-quantized ModelOpt-style checkpoints should not spill into
vision/audio encoders that have no corresponding scale tensors.

## Ownership Boundary

- Upstream `vllm` owns generic quantization configs, kernels, loader semantics,
  hardware capability rules, and generic AR quantization methods.
- `vllm-omni` owns unified config routing, diffusion-specific wrappers,
  component scoping, GGUF or ModelOpt checkpoint adapters, model-specific
  prefix mapping, docs, examples, and validation.

If a new method needs missing generic kernels or loader behavior, fix upstream
`vllm` first. In `vllm-omni`, add thin integration and model wiring.

## Working Loop

1. Read the local method doc under `docs/user_guide/quantization/`.
2. Inspect the active factory and config class in `vllm_omni/quantization/`.
3. Inspect the target model pipeline and transformer for stable prefixes and
   whether every relevant vLLM linear layer receives `quant_config`.
4. For pre-quantized checkpoints, inspect `transformer/config.json` and the
   checkpoint tensor names before running full generation.
5. Validate with the same prompt, seed, size, scheduler, steps, dtype, eager or
   non-eager mode, and parallelism as the BF16 baseline.
6. Report quality, latency, and memory separately. Do not call a method
   supported until quality is checked.

## Common Mistakes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `--quantization` has no visible effect | Wrong scope or unsupported model path | Check component routing and method docs |
| Some layers stay BF16 unexpectedly | `quant_config` was not threaded into all vLLM linear layers | Audit transformer constructors and prefixes |
| Quality collapses but loading succeeds | Too many sensitive layers were quantized | Add model-specific `ignored_layers` and compare to BF16 |
| ModelOpt checkpoint loads but output is corrupted | Prefixes, packed-module mapping, scale routing, or BF16 fallback is wrong | Use `references/modelopt-fp8.md` |
| GGUF shape or tensor mismatch | Missing architecture-specific adapter | Add explicit adapter mapping; avoid generic fallback |
| Online and offline MXFP4 disagree | Wrong `mxfp4` vs `mxfp4_dualscale` mode | Check `docs/user_guide/quantization/mxfp4.md` |
| Quantized path is slower than BF16 | Kernel path, compile mode, dtype, or shape mismatch | Compare eager-to-eager and non-eager-to-non-eager |

## References

- General methods and config routing: [references/methods.md](references/methods.md)
- Modality and model scope: [references/modality-compat.md](references/modality-compat.md)
- Diffusion implementation patterns: [references/diffusion.md](references/diffusion.md)
- Adding model support: [references/adding-models.md](references/adding-models.md)
- ModelOpt FP8 conversion and loading: [references/modelopt-fp8.md](references/modelopt-fp8.md)
