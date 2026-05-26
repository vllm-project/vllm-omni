# ModelOpt Quantization

## Overview

ModelOpt quantization loads checkpoints produced by NVIDIA ModelOpt. The
quantized weights and scale tensors are generated before serving, so inference
does not run online calibration or convert a BF16 checkpoint at startup.

vLLM-Omni validates ModelOpt FP8, ModelOpt NVFP4, and ModelOpt mixed
FP8/NVFP4 checkpoint loading for diffusion transformer stages. The loader
auto-detects supported ModelOpt checkpoint configs and keeps non-transformer
components, such as the tokenizer, scheduler, text encoder, vision/audio
encoder, and VAE, on the base checkpoint unless a model-specific guide says
otherwise.

!!! note
    ModelOpt checkpoints are pre-quantized checkpoints. Do not pass
    `--quantization fp8` for these checkpoints. The checkpoint
    `quantization_config` selects the ModelOpt path.

!!! note
    `--force-cutlass-fp8`, `--linear-backend cutlass`, and
    `--moe-backend cutlass` are runtime backend selections for checkpoints that
    already carry supported ModelOpt quantized weights and scales. They do not
    quantize BF16 checkpoints at startup.

## Supported ModelOpt Checkpoint Formats

vLLM-Omni treats ModelOpt checkpoints as pre-quantized checkpoints. The
checkpoint config must identify ModelOpt as the quantization method or producer,
and the quantization algorithm must be one of the validated algorithms below.

| Checkpoint field | Supported value |
|------------------|-----------------|
| `method` / `quant_method` | `modelopt`, `modelopt_fp4`, `modelopt_mixed` |
| `producer.name` | `modelopt` |
| `quant_algo` | `FP8`, `FP8_PER_CHANNEL_PER_TOKEN`, `NVFP4`, `MIXED_PRECISION` |

| `quant_algo` | Runtime method | Typical use |
|--------------|----------------|-------------|
| `FP8`, `FP8_PER_CHANNEL_PER_TOKEN` | `modelopt` | FP8 diffusion transformer checkpoints |
| `NVFP4` | `modelopt_fp4` | NVFP4 diffusion transformer checkpoints |
| `MIXED_PRECISION` | `modelopt_mixed` | Mixed FP8/NVFP4 checkpoints with a ModelOpt per-layer policy |

For multi-component diffusion or omni models, only the checkpoint component
that contains ModelOpt quantized weights should use the ModelOpt quantization
method. Encoders, decoders, tokenizers, schedulers, and other BF16 components
stay unquantized unless the model-specific recipe validates otherwise.

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
The optional CUTLASS FP8 runtime override requires CUDA SM89+. ModelOpt NVFP4
and mixed FP8/NVFP4 diffusion checkpoints are currently validated on Blackwell
CUDA systems in the recipes below; other CUDA generations require separate
backend and quality validation.

## Model Type Support

### Diffusion Model

| Model | HF checkpoint | Scope | Status |
|-------|---------------|-------|--------|
| Qwen-Image 2512 | `feizhai123/qwen-image-2512-modelopt-fp8-dynamic-all` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| Qwen-Image 2512 | `feizhai123/qwen-image-2512-modelopt-mixed-fp8-sensitive-nvfp4-heavy` | Diffusion transformer | Validated for ModelOpt mixed FP8/NVFP4 checkpoints |
| Z-Image | `feizhai123/z-image-modelopt-fp8-conservative` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| FLUX.2-dev | `feizhai123/flux2-dev-modelopt-fp8` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| FLUX.2-klein 4B | `feizhai123/flux2-klein-4b-modelopt-fp8` | Diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| HunyuanImage-3.0 | `feizhai123/hunyuan-image3-modelopt-fp8` | MoE diffusion transformer | Validated for ModelOpt FP8 checkpoints |
| HunyuanImage-3.0 | `feizhai123/hunyuan-image3-modelopt-mixed-experts-nvfp4-dense-fp8` | MoE diffusion transformer | Validated for ModelOpt mixed FP8/NVFP4 checkpoints |
| Wan2.2 | Not available | Diffusion transformer | Not validated |

For full serving commands and benchmark context, see
[`recipes/Qwen/Qwen-Image.md`](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen-Image.md)
and
[`recipes/Tencent/HunyuanImage-3.0-Instruct.md`](https://github.com/vllm-project/vllm-omni/blob/main/recipes/Tencent/HunyuanImage-3.0-Instruct.md).

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

For pre-quantized ModelOpt checkpoints, no `--quantization fp8` flag is needed.
The checkpoint config selects the ModelOpt path.

Online serving:

```bash
vllm serve <modelopt-checkpoint> \
  --omni \
  --tensor-parallel-size <N> \
  --linear-backend cutlass \
  --force-cutlass-fp8
```

For mixed FP8/NVFP4 MoE checkpoints, also select the validated MoE backend:

```bash
vllm serve <modelopt-mixed-moe-checkpoint> \
  --omni \
  --tensor-parallel-size <N> \
  --enable-expert-parallel \
  --linear-backend cutlass \
  --moe-backend cutlass \
  --force-cutlass-fp8
```

Offline inference:

```bash
python examples/offline_inference/text_to_image/text_to_image.py \
  --model <modelopt-checkpoint> \
  --tensor-parallel-size <N> \
  --prompt "a red ceramic teapot on a wooden table" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 20 \
  --seed 42 \
  --output outputs/modelopt.png
```

Python API:

```python
from vllm_omni import Omni

omni = Omni(
    model="<modelopt-checkpoint>",
    tensor_parallel_size=2,
    force_cutlass_fp8=True,
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_cutlass_fp8` / `--force-cutlass-fp8` | bool | `False` | Force CUTLASS FP8 linear kernels for supported ModelOpt FP8 diffusion stages on CUDA SM89+ |
| `--linear-backend cutlass` | str | auto | Select the validated CUTLASS linear backend for supported ModelOpt NVFP4 or mixed FP8/NVFP4 diffusion stages |
| `--moe-backend cutlass` | str | auto | Select the validated CUTLASS MoE backend for supported ModelOpt mixed MoE checkpoints |

## Validation and Notes

1. Compare the ModelOpt checkpoint against the BF16 baseline with the same
   prompt, resolution, seed, and inference steps.
2. Use `tests/diffusion/quantization/test_quantization_quality.py` with
   `VLLM_OMNI_QUALITY_CONFIGS` to validate local baseline and quantized model
   paths.
3. For HunyuanImage-3.0 quantized DiT checkpoints, the opt-in accuracy check is:

   ```bash
   CUDA_VISIBLE_DEVICES=2,3 \
   HUNYUAN_IMAGE3_RUN_QUANT_ACCURACY=1 \
   HUNYUAN_IMAGE3_QUANT_DEVICES=0,1 \
   HUNYUAN_IMAGE3_QUANT_TP=2 \
   HUNYUAN_IMAGE3_BF16_MODEL=/path/to/hunyuan-image3-bf16 \
   HUNYUAN_IMAGE3_FP8_MODEL=/path/to/hunyuan-image3-modelopt-fp8 \
   HUNYUAN_IMAGE3_NVFP4_MODEL=/path/to/hunyuan-image3-modelopt-mixed-experts-nvfp4-dense-fp8 \
   PYTHONPATH=/path/to/vllm-omni:${PYTHONPATH:-} \
   python -m pytest -s -v \
     tests/e2e/accuracy/test_hunyuan_image3.py \
     -k quantized_dit_matches_bf16_accuracy
   ```

4. Report CLIP score deltas, SSIM, PSNR, throughput, latency, and peak memory
   when adding a new validated ModelOpt diffusion checkpoint.
5. Keep `--quantization fp8` for online FP8 from BF16 checkpoints; use this
   ModelOpt path only when the checkpoint already contains ModelOpt quantized
   weights and scales.
