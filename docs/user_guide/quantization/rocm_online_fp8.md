# Online FP8 block scaling on ROCm

ROCm can opt into finer FP8 weight and activation scaling when per-tensor
scales lose too much diffusion quality. Set `online_block_size` to 32, 64 or
128 while loading an ordinary BF16/FP16 checkpoint:

```python
from vllm_omni import Omni

omni = Omni(
    model="Tongyi-MAI/Z-Image-Turbo",
    quantization_config={
        "method": "fp8",
        "online_block_size": 64,
        "ignored_layers": ["img_mlp"],
    },
)
```

The option uses square weight blocks and dynamic activation groups of
`1 × online_block_size`. Existing layer exclusions and component routing still
apply. It requires ROCm and dynamic activation scaling; serialized FP8
checkpoints keep their own `weight_block_size` metadata. Validate each model's
quality against BF16 with its complete layer policy before selecting a block
size. The Z-Image ROCm CI recipe uses 64 with its existing sensitive layers
excluded; the NVIDIA recipe keeps its original scales.

The example only illustrates configuration. Use the complete model-specific
layer policy for quality-sensitive production workloads. See
[FP8 quantization](fp8.md) for component scope and checkpoint configuration.
