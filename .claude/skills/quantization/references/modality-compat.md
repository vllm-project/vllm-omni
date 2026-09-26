# Quantization Compatibility by Modality

Use this as a routing guide. The source of truth is the local method docs under
`docs/user_guide/quantization/` and the current model implementation.

## Summary

- AR and omni language-model quantization mostly follows upstream `vllm`, with
  vLLM-Omni scoping for multi-stage models.
- Diffusion quantization is method-specific and model-specific. It usually
  targets the diffusion transformer only.
- Pre-quantized checkpoints should quantize only the component that has
  quantized weights and scale tensors.

## Model Type Matrix

| Model Type | Typical Quantization Scope | Notes |
|------------|----------------------------|-------|
| AR text models | Language model | Upstream `vllm` methods usually apply |
| Omni/TTS models | Thinker or AR language-model stage | Audio, vision, talker, and codec stages usually stay BF16 |
| DiT image models | Diffusion transformer | Check FP8, Int8, GGUF, ModelOpt, AutoRound docs |
| DiT video models | Diffusion transformer | HunyuanVideo FP8 and Wan2.2 MXFP paths are documented; validate other video paths |
| Multi-stage diffusion | Selected stage only | Use per-component routing and per-stage validation |
| Audio diffusion | Usually unsupported | Prefer dtype, offload, cache, or parallelism unless docs say otherwise |

## Method Hints

| Goal | Candidate |
|------|-----------|
| AR memory reduction | AWQ, GPTQ, AutoRound, ModelOpt where checkpoint supports it |
| CUDA diffusion W8A8 | FP8 or Int8 |
| Pre-quantized CUDA diffusion checkpoint | ModelOpt or GGUF |
| Ascend NPU diffusion | Int8, MXFP8, MXFP4, MXFP4 DualScale, msModelSlim |
| Intel XPU checkpoint | AutoRound/INC paths |
| Conservative image quality | selective FP8/Int8 with `ignored_layers` |
| Unsupported diffusion model | cache acceleration, CPU/layerwise offload, dtype, TP/SP/CFG parallelism |

## Validation Rule

Compatibility means more than successful loading. A method is compatible only
after it has:

- correct component scope
- complete checkpoint/config metadata if offline
- fixed-seed quality comparison against BF16
- acceptable latency and memory behavior
- docs or examples that state model, method, hardware, and caveats
