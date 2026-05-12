# ModelOpt Quantization

## Overview

ModelOpt quantization loads checkpoints produced by NVIDIA ModelOpt. The
quantized weights and scale tensors are generated before serving, so inference
does not run online calibration or convert a BF16 checkpoint at startup.

vLLM-Omni currently validates the ModelOpt FP8 checkpoint path for diffusion
transformers. The loader auto-detects supported ModelOpt FP8 checkpoint configs
and keeps non-transformer components, such as the tokenizer, scheduler, text
encoder, and VAE, on the base checkpoint unless a model-specific guide says
otherwise.

!!! note
    `--force-cutlass-fp8` is an explicit runtime override for diffusion
    checkpoints that already carry a supported ModelOpt FP8 config. It does not
    quantize BF16 checkpoints and it does not apply to online `--quantization
    fp8`. The flag only takes effect for ModelOpt FP8 diffusion stages on CUDA
    SM89+ devices; other platforms and non-ModelOpt FP8 paths fall back to the
    normal vLLM kernel selection.

## Supported ModelOpt Checkpoint Formats

vLLM-Omni treats ModelOpt checkpoints as pre-quantized checkpoints. The
checkpoint config must identify ModelOpt as the quantization method or producer,
and the quantization algorithm must be one of the validated FP8 algorithms.

| Checkpoint field | Supported value |
|------------------|-----------------|
| `method` / `quant_method` | `modelopt` |
| `producer.name` | `modelopt` |
| `quant_algo` | `FP8`, `FP8_PER_CHANNEL_PER_TOKEN` |

Other ModelOpt algorithms, such as NVFP4, are not enabled by this diffusion
FP8 path until they have separate model and quality validation.

## Hardware Support

| Device | Support |
|--------|---------|
| NVIDIA Blackwell GPU (SM 100+) | ✅ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ |
| NVIDIA Ampere GPU (SM 80+) | ⭕ |
| AMD ROCm | ⭕ |
| Intel XPU | ⭕ |
| Ascend NPU | ❌ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this guide.
The optional CUTLASS FP8 runtime override requires CUDA SM89+.

## Model Type Support

### Diffusion Model

| Model | HF checkpoint | Scope | Status |
|-------|---------------|-------|--------|
| Qwen-Image 2512 | `feizhai123/qwen-image-2512-modelopt-fp8-dynamic-all` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| Z-Image | `feizhai123/z-image-modelopt-fp8-conservative` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| FLUX.2-dev | `feizhai123/flux2-dev-modelopt-fp8` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| FLUX.2-klein 4B | `feizhai123/flux2-klein-4b-modelopt-fp8` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| HunyuanImage-3.0 | `feizhai123/hunyuan-image3-modelopt-fp8` | MoE diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| Wan2.2 | Not available | Diffusion transformer | Not validated |

### Multi-Stage Omni/TTS Model

| Model | Scope | Status |
|-------|-------|--------|
| Qwen3-Omni | Thinker language-model stage | ModelOpt FP8 checkpoint path |
| Qwen3-TTS | TTS language-model stage | Not validated |

Audio encoder, vision encoder, talker, and code2wav stages stay in BF16 unless
a model-specific guide documents otherwise.

### Multi-Stage Diffusion Model

ModelOpt checkpoints must be routed to the stage whose checkpoint contains the
ModelOpt `quantization_config`. BAGEL and GLM-Image are not listed as validated
ModelOpt targets yet.

## Configuration

For pre-quantized ModelOpt FP8 checkpoints, no `--quantization fp8` flag is
needed. The checkpoint config selects the ModelOpt path.

Online serving:

```bash
vllm serve <modelopt-fp8-checkpoint> \
  --omni \
  --tensor-parallel-size <N> \
  --force-cutlass-fp8
```

Offline inference:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model <modelopt-fp8-checkpoint> \
  --tensor-parallel-size <N> \
  --prompt "a red ceramic teapot on a wooden table" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 20 \
  --seed 42 \
  --output outputs/modelopt_fp8.png
```

Python API:

```python
from vllm_omni import Omni

omni = Omni(
    model="<modelopt-fp8-checkpoint>",
    tensor_parallel_size=2,
    force_cutlass_fp8=True,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_cutlass_fp8` / `--force-cutlass-fp8` | bool | `False` | Force CUTLASS FP8 linear kernels for supported ModelOpt FP8 diffusion stages on CUDA SM89+ |

## Validation and Notes

1. Compare the ModelOpt FP8 checkpoint against the BF16 baseline with the same
   prompt, resolution, seed, and inference steps.
2. Use `tests/diffusion/quantization/test_quantization_quality.py` with
   `VLLM_OMNI_QUALITY_CONFIGS` to validate local baseline and quantized model
   paths.
3. Report LPIPS, PSNR, MAE, throughput, latency, and peak memory when adding a
   new validated ModelOpt diffusion checkpoint.
4. Keep `--quantization fp8` for online FP8 from BF16 checkpoints; use this
   ModelOpt path only when the checkpoint already contains ModelOpt FP8 weights
   and scales.
