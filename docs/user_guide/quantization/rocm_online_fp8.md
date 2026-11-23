# Online FP8 Block Scaling on ROCm

ROCm can opt into finer online FP8 groups when per-tensor or fixed 128-block
scaling loses too much diffusion quality. Set `online_block_size` to `32`,
`64`, or `128` while loading an ordinary BF16/FP16 checkpoint:

```python
from vllm_omni import Omni

omni = Omni(
    model="<your-model>",
    quantization_config={
        "method": "fp8",
        "online_block_size": 64,
        "ignored_layers": ["<model-specific-sensitive-layer>"],
    },
)
```

Serving accepts the same configuration as JSON:

```bash
vllm serve <your-model> --omni \
  --diffusion-quantization-config \
  '{"method":"fp8","online_block_size":64,"ignored_layers":["<model-specific-sensitive-layer>"]}'
```

The option uses square `online_block_size × online_block_size` weight groups
and dynamic `1 × online_block_size` activation groups. Existing component
routing and `ignored_layers` still apply. The option is ROCm-only, requires
dynamic activation scaling, and cannot be combined with serialized FP8
checkpoint metadata or `weight_block_size`.

Z-Image's ROCm quality configuration uses `64` together with its complete
checked-in sensitive-layer exclusions. Do not copy only part of that layer
policy into production. Validate every model and block size against a BF16
baseline with the same prompt, seed, resolution, and sampling settings.

The AITER block kernel currently accepts `1 × 128` activation groups. Finer
groups can select vLLM's ROCm Triton block kernel instead, so measure latency
and throughput as well as quality. Configurable online blocks are not yet
supported with the Diffusers backend, DLO AllGather, or HSDP.

See [FP8 quantization](fp8.md) for component scope, ignored-layer guidance,
and serialized-checkpoint configuration.
