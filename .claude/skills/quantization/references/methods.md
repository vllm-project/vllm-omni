# General Quantization Methods

Use this file for method selection and unified `vllm-omni` config syntax. For
diffusion-specific wiring, see `diffusion.md`. For ModelOpt checkpoint export
and loading details, see `modelopt-fp8.md`.

## Unified Entry Point

`vllm_omni.quantization.build_quant_config()` accepts:

- `None` or `"none"` to disable quantization
- a method string such as `"fp8"`, `"int8"`, `"mxfp8"`, `"awq"`, or `"auto-round"`
- a flat dict with `"method"` or `"quant_method"`
- a per-component dict such as `{"transformer": {"method": "fp8"}, "vae": None}`
- an already-built `QuantizationConfig`

Per-component dicts are routed by `ComponentQuantizationConfig` using
longest-prefix matching. Prefix stability matters: if a model remaps HF names
to vLLM names, apply the mapper to `ignored_layers` or component prefixes.

## Local Overrides

The local factory delegates most methods to upstream `vllm`, but has vLLM-Omni
overrides for methods that need diffusion, NPU, XPU, or component behavior:

| Method | Local Config | Main Use |
|--------|--------------|----------|
| `gguf` | `DiffusionGGUFConfig` | Pre-quantized diffusion transformer weights |
| `int8` | `DiffusionInt8Config` | Online or serialized W8A8 diffusion transformers |
| `mxfp8` | `DiffusionMXFP8Config` | W8A8 MXFP8 diffusion on NPU and online XPU path |
| `mxfp4` | `DiffusionMXFP4Config` | Online W4A4 MXFP4 diffusion on NPU |
| `mxfp4_dualscale` | `DiffusionMXFP4DualScaleMixedConfig` | Online/offline W4A4 dual-scale plus BF16 fallback on NPU |
| `inc`, `auto-round`, `auto_round` | `OmniINCConfig` | AutoRound/INC checkpoint loading, including MXFP8 checkpoint metadata |
| ModelOpt config dicts | upstream ModelOpt configs through auto-detect | FP8, NVFP4, or mixed ModelOpt checkpoints |

Check `SUPPORTED_QUANTIZATION_METHODS` in `vllm_omni/quantization/factory.py`
before adding a new method name.

## AR and Generic Quantization

For AR-backed models, upstream `vllm` methods usually apply:

- `awq` and `gptq`: 4-bit or 8-bit weight quantization with calibration,
  commonly used for text and AR stages.
- `fp8`: FP8 weight path or KV-cache FP8 depending on flags and hardware.
- `inc` or `auto-round`: pre-quantized checkpoints with Intel Neural
  Compressor / AutoRound metadata.

Use these for the AR language model stage only when the checkpoint and model
guide support it. Omni and TTS models often keep audio, vision, talker, and
codec stages in BF16.

## Diffusion and Offline Methods

For diffusion transformers, prefer the method-specific docs:

- `docs/user_guide/quantization/fp8.md`
- `docs/user_guide/quantization/int8.md`
- `docs/user_guide/quantization/gguf.md`
- `docs/user_guide/quantization/modelopt.md`
- `docs/user_guide/quantization/mxfp8.md`
- `docs/user_guide/quantization/mxfp4.md`
- `docs/user_guide/quantization/autoround.md`
- `docs/user_guide/quantization/msmodelslim.md`

Pre-quantized checkpoints should normally auto-detect from config. Runtime
flags such as `--force-cutlass-fp8`, `--linear-backend cutlass`, and
`--moe-backend cutlass` select kernels; they do not quantize a BF16 checkpoint.

## Validation

Use fixed-seed A/B comparisons and record:

- model path and quantized checkpoint path
- method config and `ignored_layers`
- prompt, size, frame count, steps, scheduler, dtype, and seed
- eager/non-eager mode, compile flags, and parallelism
- output metrics such as PSNR, MAE, cosine similarity, SSIM, LPIPS, plus visual artifacts
- latency and peak memory, reported separately from quality

The helper `vllm_omni.quantization.tools.compare_diffusion_trajectory_similarity`
is the default lightweight image/video comparison tool for diffusion paths.
