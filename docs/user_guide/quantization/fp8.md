# FP8 Quantization

## Overview

FP8 quantization converts BF16/FP16 weights to FP8 at model load time. Online
activation scaling is the default and does not require calibration. Static
activation scaling is supported when calibrated scale information is available.
For ModelOpt-produced pre-quantized checkpoints, see
[ModelOpt Quantization](modelopt.md).

Some architectures can quantize all linear layers. Others have
quality-sensitive layers that should stay in BF16 through `ignored_layers`.
Image-stream MLPs (`img_mlp`) are a common sensitive target because denoising
latent ranges shift across timesteps and small per-layer errors can compound
in deep DiT blocks.

## Hardware Support

| Device | Support |
|--------|---------|
| NVIDIA Blackwell GPU (SM 100+) | ✅ |
| NVIDIA Ada/Hopper GPU (SM 89+) | ✅ |
| NVIDIA Ampere GPU (SM 80+) | ✅ |
| AMD ROCm | ⭕ |
| Intel XPU | ⭕ |
| Ascend NPU | ❌ |

Legend: `✅` supported, `❌` unsupported, `⭕` not verified in this
guide. FP8 on Ampere may use a weight-only path where available.

### Faster FP8 GEMM on Blackwell (quack)

On Blackwell (SM 100+), vLLM runs FP8 linears through the FlashInfer kernel, which
applies the bias as a separate kernel after the GEMM. On the small GEMMs in video
DiTs this bias add is a significant overhead. Installing the optional `quack` kernel
lets vLLM-Omni fuse `alpha * (A @ B) + bias` into a single CuteDSL GEMM, recovering
that overhead (e.g. HunyuanVideo-1.5 FP8 goes from slower-than-BF16 to faster).

```bash
# CUDA 12.9
pip install vllm-omni[quack]

# CUDA 13.x
pip install 'quack-kernels[cu13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

It is enabled automatically once installed (no flag needed) and is **Blackwell-only**:
on Hopper/Ada the CUTLASS FP8 kernel already fuses bias, so quack is not used there.
Set `VLLM_OMNI_USE_QUACK_FP8=0` to force the FlashInfer path. If `quack-kernels` is
not installed, FP8 still works — it just keeps the unfused FlashInfer path.

#### Compile cache and warmup

quack JIT-compiles its kernel once per distinct GEMM shape (tens of seconds, longer
the first time across all autotuned configs). The compiled `.o` files are cached on
disk and reused on later runs, so this is a one-time cost — **not** per request.

vLLM-Omni points that cache at `~/.cache/vllm_omni/quack` (override with
`QUACK_CACHE_DIR`) instead of quack's default under `/tmp`, so it survives restarts.
In containers, set `QUACK_CACHE_DIR` to a mounted/persistent path — or bake it into
the image — so the first cold start does not recompile. The engine's startup dummy
run already exercises the kernels, so with a warm cache the first real request is fast.

To pre-warm specific shapes (e.g. at image build time):

```python
from vllm_omni.quantization.quack_fp8 import warmup_quack_fp8
# (M, K, N) per linear; M = number of tokens for your resolution/frame count
warmup_quack_fp8([(14040, 2048, 6144), (14040, 2048, 2048)])
```

> The PyPI package is `quack-kernels` (imported as `quack`); plain `pip install
> quack` is an unrelated statistics library. Requires CUDA 12.9+ and Python 3.12.

## Model Type Support

### Diffusion Model (Qwen-Image, Wan2.2)

| Model | HF models | Online | Pre-calibrated | Recommendation | `ignored_layers` | Text-Encoder quantization |
|-------|-----------|:-------:|:------:|----------------|------------------|------------------|
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Yes | Yes | Skip sensitive image-stream MLPs when quality regresses | `img_mlp` | |
| Wan2.2 | Wan2.2 diffusion pipelines | Not validated | Not validated | Validate against BF16 before documenting as supported | TBD | |
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | Yes | Yes | All layers | None | ✅︎ |
| FLUX.1 | `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-schnell` | Yes | Yes | All layers | None | |
| FLUX.2-klein | `black-forest-labs/FLUX.2-klein-4B` | Yes | Yes | All layers | None | |
| HunyuanImage-3.0 | `tencent/HunyuanImage-3.0`, `tencent/HunyuanImage-3.0-Instruct` | Yes | Yes | All layers; use the Hunyuan stage config for multi-stage runs | None | |
| HunyuanVideo-1.5 | `hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`, `720p_t2v`, `480p_i2v` | Yes | Yes | All layers | None | |

### Multi-Stage Omni/TTS Model (Qwen3-Omni, Qwen3-TTS)

| Model | Scope | Format | Status |
|-------|-------|--------|--------|
| Qwen3-Omni | Thinker language-model stage | [ModelOpt](modelopt.md) `quant_algo=FP8` | Tested for thinker memory reduction |
| Qwen3-TTS | TTS language-model stage | Checkpoint config | Not validated |

Audio encoder, vision encoder, talker, and code2wav stay in BF16 unless a
model-specific guide says otherwise.

### Multi-Stage Diffusion Model (BAGEL, GLM-Image)

| Model | Scope | Status | Notes |
|-------|-------|--------|-------|
| BAGEL | Stage-specific transformer or DiT module | Not validated | Route FP8 to the intended stage before enabling |
| GLM-Image | Stage-specific transformer or DiT module | Not validated | Validate quality against BF16 baseline |

## Configuration

Python API:

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(model="<your-model>", quantization="fp8")

omni_with_skips = Omni(
    model="<your-model>",
    quantization_config={
        "method": "fp8",
        "ignored_layers": ["img_mlp"],
    },
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

CLI:

```bash
python text_to_image.py --model <your-model> --quantization fp8
python text_to_image.py --model <your-model> --quantization fp8 --ignored-layers "img_mlp"
vllm serve <your-model> --omni --quantization fp8
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | - | Quantization method (`"fp8"`) |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16 |
| `activation_scheme` | str | `"dynamic"` | `"dynamic"` selects online activation scaling, or `"static"` when scales are available |
| `weight_block_size` | list[int] \| None | `None` | Block size for block-wise weight quantization |

The available `ignored_layers` names depend on the model architecture, for
example `to_qkv`, `to_out`, `img_mlp`, or `txt_mlp`.

## Validation and Notes

FP8 quantization can be combined with cache acceleration:

```python
omni = Omni(
    model="<your-model>",
    quantization="fp8",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```

Compare generated outputs with a BF16 baseline before adding a new model to the
supported table. GLM-Image and Helios are not listed as FP8-supported diffusion
models until they have method-specific validation.
