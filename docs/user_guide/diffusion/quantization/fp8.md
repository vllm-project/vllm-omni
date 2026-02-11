# FP8 Quantization

## Overview

FP8 quantization converts BF16/FP16 weights to FP8 at model load time. No calibration or pre-quantized checkpoint needed.

Depending on the model, either all layers can be quantized, or some sensitive layers should stay in BF16. See the [per-model table](#supported-models) for which case applies.

Common sensitive layers in DiT-based diffusion models include **image-stream MLPs** (`img_mlp`). These are particularly vulnerable to FP8 precision loss because they process denoising latents whose dynamic range shifts significantly across timesteps, and unlike attention projections (which benefit from QK-Norm stabilization), MLPs have no built-in normalization to absorb quantization error. In deep architectures (e.g., 60+ residual blocks), small per-layer errors compound and degrade output quality. Other layers such as **attention projections** (`to_qkv`, `to_out`) and **text-stream MLPs** (`txt_mlp`) are generally more robust due to normalization or more stable input statistics.

## Configuration

1. **Python API**: set `quantization="fp8"`. To skip sensitive layers, use `quantization_config` with `ignored_layers`.

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# All layers quantized
omni = Omni(model="<your-model>", quantization="fp8")

# Skip sensitive layers
omni = Omni(
    model="<your-model>",
    quantization_config={
        "method": "fp8",
        "ignored_layers": ["<layer-name>"],
    },
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

2. **CLI**: pass `--quantization fp8` and optionally `--ignored-layers`.

```bash
# All layers
python text_to_image.py --model <your-model> --quantization fp8

# Skip sensitive layers
python text_to_image.py --model <your-model> --quantization fp8 --ignored-layers "img_mlp"

# Online serving
vllm serve <your-model> --omni --quantization fp8
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | Quantization method (`"fp8"`) |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16 |
| `activation_scheme` | str | `"dynamic"` | `"dynamic"` (no calibration) or `"static"` |
| `weight_block_size` | list[int] \| None | `None` | Block size for block-wise weight quantization |

The available `ignored_layers` names depend on the model architecture (e.g., `to_qkv`, `to_out`, `img_mlp`, `txt_mlp`). Consult the transformer source for your target model.

## Supported Models

| Model | HF Models | Recommendation | `ignored_layers` |
|-------|-----------|---------------|------------------|
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | All layers | None |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | Skip sensitive layers | `img_mlp` |

## Combining with Other Features

FP8 quantization can be combined with cache acceleration:

```python
omni = Omni(
    model="<your-model>",
    quantization="fp8",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```
