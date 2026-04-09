# Int8 Quantization

## Overview

Int8 quantization converts BF16/FP16 weights to Int8 at model load time. No calibration or pre-quantized checkpoint needed.

Depending on the model, either all layers can be quantized, or some sensitive layers should stay in BF16/FP16. See the [per-model table](#supported-models) for which case applies.

## Configuration

1. **Python API**: set `quantization="int8"`. To skip sensitive layers, use `quantization_config` with `ignored_layers`.

```python
from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# All layers quantized
omni = Omni(model="<your-model>", quantization="int8")

# Skip sensitive layers
omni = Omni(
    model="<your-model>",
    quantization_config={
        "method": "int8",
        "ignored_layers": ["<layer-name>"],
    },
)

outputs = omni.generate(
    "A cat sitting on a windowsill",
    OmniDiffusionSamplingParams(num_inference_steps=50),
)
```

2. **CLI**: pass `--quantization int8` and optionally `--ignored-layers`.

```bash
# All layers
python text_to_image.py --model <your-model> --quantization int8

# Skip sensitive layers
python text_to_image.py --model <your-model> --quantization int8 --ignored-layers "img_mlp"

# Online serving
vllm serve <your-model> --omni --quantization int8
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | str | — | Quantization method (`"int8"`) |
| `ignored_layers` | list[str] | `[]` | Layer name patterns to keep in BF16/FP16 |
| `activation_scheme` | str | `"dynamic"` | `"dynamic"` (no calibration) |


The available `ignored_layers` names depend on the model architecture (e.g., `to_qkv`, `to_out`, `img_mlp`, `txt_mlp`). Consult the transformer source for your target model.

## Supported Models

| Model | HF Models | Recommendation | `ignored_layers` |
|-------|-----------|---------------|------------------|
| Z-Image | `Tongyi-MAI/Z-Image-Turbo` | All layers | None |
| Qwen-Image | `Qwen/Qwen-Image`, `Qwen/Qwen-Image-2512` | All layers | None |

## Combining with Other Features

Int8 quantization can be combined with cache acceleration:

```python
omni = Omni(
    model="<your-model>",
    quantization="int8",
    cache_backend="tea_cache",
    cache_config={"rel_l1_thresh": 0.2},
)
```
